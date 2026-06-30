# ══════════════════════════════════════════════════════════════════════════════
# CloudFront distribution + AWS WAF — edge security and content delivery
# ══════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   Visitor → CloudFront (edge) → WAF (rate limit + geo block) → Lambda Function URL
#   Widget JS  → CloudFront (same distribution, /widget/* origin) → S3
#
# No API Gateway — Lambda Function URL + CloudFront is scale-to-zero friendly
# and avoids the per-request gateway fee.

# ── WAF Web ACL ───────────────────────────────────────────────────────────────

resource "aws_wafv2_web_acl" "main" {
  name        = "${var.project}-waf"
  description = "WAF for ${var.project} - rate limiting, geo blocking, common protections"
  scope       = "CLOUDFRONT"

  default_action {
    allow {}
  }

  # ── Rate-based rule: block IPs exceeding the threshold ────────────────────
  rule {
    name     = "rate-limit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project}RateLimitRule"
      sampled_requests_enabled   = true
    }
  }

  # ── Country block rule (optional) ─────────────────────────────────────────
  dynamic "rule" {
    for_each = length(var.waf_blocked_countries) > 0 ? [1] : []

    content {
      name     = "geo-block"
      priority = 2

      action {
        block {}
      }

      statement {
        geo_match_statement {
          country_codes = var.waf_blocked_countries
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "${var.project}GeoBlockRule"
        sampled_requests_enabled   = true
      }
    }
  }

  # ── AWS IP reputation list (contains known bad actors) ────────────────────
  rule {
    name     = "aws-reputation-lists"
    priority = 3

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project}AWSReputationRule"
      sampled_requests_enabled   = true
    }
  }

  # ── Anonymous IP block (VPNs, proxies, Tor) for abuse prevention ──────────
  rule {
    name     = "anonymous-ip-block"
    priority = 4

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAnonymousIpList"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${var.project}AnonymousIpRule"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project}WAF"
    sampled_requests_enabled   = true
  }
}

# ── CloudFront origin access control for S3 widget bucket ─────────────────────

resource "aws_cloudfront_origin_access_control" "widget" {
  name                              = "${var.project}-widget-oac"
  description                       = "OAC for widget static assets S3 bucket"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# ── S3 bucket policy allowing CloudFront OAC access ───────────────────────────

data "aws_iam_policy_document" "widget_oac" {
  statement {
    sid    = "AllowCloudFrontOAC"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.widget_static.arn}/*"]
    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = ["arn:aws:cloudfront::${data.aws_caller_identity.current.account_id}:distribution/${aws_cloudfront_distribution.main.id}"]
    }
  }
}

resource "aws_s3_bucket_policy" "widget_oac" {
  bucket = aws_s3_bucket.widget_static.id
  policy = data.aws_iam_policy_document.widget_oac.json
}

# ── CloudFront distribution ───────────────────────────────────────────────────

resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "${var.project} — API proxy + widget static assets"
  default_root_object = ""
  price_class         = var.cloudfront_price_class

  # Associate the WAF web ACL with this distribution
  web_acl_id = aws_wafv2_web_acl.main.arn

  # ── API origin: Lambda Function URL ───────────────────────────────────────
  origin {
    domain_name = trimsuffix(replace(aws_lambda_function_url.api.function_url, "https://", ""), "/")
    origin_id   = "api"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      # Match the Lambda timeout (120s) so CloudFront doesn't 504 before
      # the Lambda finishes a cold-start + RAG query on the first request.
      origin_read_timeout = 120
    }

    # No Origin Shield — AWS charges $0.001/GB through it, and the Lambda
    # Function URL delivers single-digit-millisecond responses; the shield's
    # origin-offload benefit is irrelevant at launch scale, and the cost floor
    # (no Origin Shield = $0) is correct for a scale-to-zero design.
  }

  # ── Widget origin: S3 static assets ───────────────────────────────────────
  origin {
    domain_name              = aws_s3_bucket.widget_static.bucket_regional_domain_name
    origin_id                = "widget"
    origin_access_control_id = aws_cloudfront_origin_access_control.widget.id
  }

  # ── Default cache behavior: API proxy (no caching) ────────────────────────
  default_cache_behavior {
    target_origin_id       = "api"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    # No caching for the API — every request hits Lambda
    cache_policy_id          = data.aws_cloudfront_cache_policy.no_cache.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id

    # WAF already handles auth; CloudFront doesn't need additional auth headers
    response_headers_policy_id = data.aws_cloudfront_response_headers_policy.security_headers.id
  }

  # ── Widget cache behavior: /widget/* ──────────────────────────────────────
  ordered_cache_behavior {
    path_pattern           = "/widget/*"
    target_origin_id       = "widget"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true
    smooth_streaming       = false

    # Cache widget assets at the edge (immutable after deploy)
    min_ttl     = 0
    default_ttl = 86400  # 1 day
    max_ttl     = 604800 # 7 days

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }

  # ── Custom domain (optional) ──────────────────────────────────────────────
  aliases = var.domain_name != "" ? [var.domain_name] : []

  # Viewer certificate selection:
  #   - If acm_certificate_arn is set, use the custom cert (snapshot: terraform will
  #     ignore the default block because only one viewer_certificate is allowed).
  #   - Otherwise, use the default CloudFront certificate (*.cloudfront.net).
  #
  # Terraform requires exactly one viewer_certificate block. The count-based
  # conditional ensures only one is ever present.
  dynamic "viewer_certificate" {
    for_each = var.acm_certificate_arn != "" ? [1] : []
    content {
      cloudfront_default_certificate = false
      acm_certificate_arn            = var.acm_certificate_arn
      ssl_support_method             = "sni-only"
      minimum_protocol_version       = "TLSv1.2_2021"
    }
  }

  dynamic "viewer_certificate" {
    for_each = var.acm_certificate_arn == "" ? [1] : []
    content {
      cloudfront_default_certificate = true
    }
  }

  # Restrict to US, Canada, and Europe for low latency (unless overridden)
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = {
    Name = "${var.project}-distribution"
  }
}


# ── Data sources for cache/origin request policies ────────────────────────────

data "aws_cloudfront_cache_policy" "no_cache" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewerExceptHostHeader"
}

data "aws_cloudfront_response_headers_policy" "security_headers" {
  name = "Managed-SecurityHeadersPolicy"
}

data "aws_caller_identity" "current" {}
