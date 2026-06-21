variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name prefix for all resource names"
  type        = string
  default     = "appther-chatbot"
}

variable "bucket_suffix" {
  description = "Globally-unique suffix for S3 bucket names (use your 12-digit AWS account ID)"
  type        = string
}

variable "budget_alert_email" {
  description = "Email address that receives AWS Budget cost alerts"
  type        = string
}

variable "secret_recovery_window_days" {
  description = "Recovery window for Secrets Manager secrets (0 in dev, 7-30 in prod)"
  type        = number
  default     = 7
}

variable "github_repo" {
  description = "GitHub repository in 'org/repo' format for OIDC role assumption"
  type        = string
  default     = "Vansh-Sharma27/appther-chatbot"
}

variable "github_oidc_provider_arn" {
  description = "ARN of the GitHub OIDC identity provider in AWS IAM. Set after creating the provider."
  type        = string
  default     = ""
}

# ── Lambda / API ──────────────────────────────────────────────────────────────

variable "api_image_tag" {
  description = "Tag of the API Docker image in ECR (e.g. 'latest' or a commit SHA)"
  type        = string
  default     = "latest"
}

variable "api_lambda_timeout" {
  description = "Lambda function timeout in seconds (30s Gemini timeout + cold start + retrieval)"
  type        = number
  default     = 120
}

variable "api_lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 1024
}

# ── WAF ───────────────────────────────────────────────────────────────────────

variable "waf_rate_limit" {
  description = "Maximum requests per 5-minute window per IP from a single IP before WAF rate-based rule blocks it"
  type        = number
  default     = 500
}

variable "waf_blocked_countries" {
  description = "List of 2-letter ISO country codes to block at the WAF level"
  type        = list(string)
  default     = []
}

# ── CloudFront ────────────────────────────────────────────────────────────────

variable "cloudfront_price_class" {
  description = "CloudFront price class: PriceClass_100 (US+EU), PriceClass_200 (US+EU+Asia), PriceClass_All"
  type        = string
  default     = "PriceClass_100"
}

variable "domain_name" {
  description = "Custom domain for the API (e.g. api.appther.com). Leave empty to use CloudFront default domain."
  type        = string
  default     = ""
}

variable "acm_certificate_arn" {
  description = "ARN of the ACM certificate in us-east-1 for the custom domain. Required if domain_name is set."
  type        = string
  default     = ""
}
