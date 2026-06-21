# ── API Lambda ECR repository ──────────────────────────────────────────────────

resource "aws_ecr_repository" "api" {
  name                 = "${var.project}/api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ── Lambda ECR pull policy (attach to existing lambda role) ────────────────────

data "aws_iam_policy_document" "lambda_ecr_pull" {
  statement {
    sid    = "ECRPull"
    effect = "Allow"
    actions = [
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [aws_ecr_repository.api.arn]
  }
}

resource "aws_iam_role_policy" "lambda_ecr" {
  name   = "${var.project}-lambda-ecr"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_ecr_pull.json
}

# ── Lambda function (container image) ──────────────────────────────────────────

resource "aws_lambda_function" "api" {
  function_name = "${var.project}-api"
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
  timeout       = var.api_lambda_timeout
  memory_size   = var.api_lambda_memory

  environment {
    variables = {
      DYNAMODB_TABLE         = aws_dynamodb_table.main.name
      SECRET_VOYAGE_ARN      = aws_secretsmanager_secret.voyage_api_key.arn
      SECRET_GEMINI_ARN      = aws_secretsmanager_secret.gemini_api_key.arn
      SECRET_JINA_ARN        = aws_secretsmanager_secret.jina_api_key.arn
      LANCE_INDEX_URI        = "s3://${aws_s3_bucket.index.bucket}/lance_db"
      CORS_ORIGINS           = "https://www.appther.com,https://appther.com"
      RATE_LIMIT             = "20/minute"
      GEMINI_LITE_MODEL      = "gemini-2.5-flash-lite"
      GEMINI_FLASH_MODEL     = "gemini-3-flash"
      GEMINI_TIMEOUT_SECONDS = "30"
      AWS_LWA_INVOKE_MODE    = "response_stream"
    }
  }

  # No VPC config — Lambda runs outside VPC to avoid NAT cost
}

# ── Lambda Function URL with response streaming ───────────────────────────────

resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_credentials = false
    allow_origins     = ["https://www.appther.com", "https://appther.com"]
    allow_methods     = ["GET", "POST"]
    allow_headers     = ["Content-Type", "X-API-Key"]
    max_age           = 3600
  }
}
