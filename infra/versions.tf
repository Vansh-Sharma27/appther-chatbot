terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state (S3 + DynamoDB locking).
  #
  # BEFORE the first terraform apply, create the state bucket and lock table
  # manually (or via a bootstrap script) and then uncomment this block with
  # the correct bucket name.  The bucket and table names are configurable via
  # terraform.tfvars but the backend block must be hardcoded at init time --
  # Terraform does not allow variable interpolation inside backend blocks.
  #
  #   aws s3 mb s3://<YOUR_TFSTATE_BUCKET> --region us-east-1
  #   aws dynamodb create-table \
  #     --table-name <YOUR_LOCK_TABLE> \
  #     --key-schema AttributeName=LockID,KeyType=HASH \
  #     --attribute-definitions AttributeName=LockID,AttributeType=S \
  #     --billing-mode PAY_PER_REQUEST
  #
  # Then init with:
  #   terraform init -backend-config="bucket=<YOUR_TFSTATE_BUCKET>" \
  #     -backend-config="dynamodb_table=<YOUR_LOCK_TABLE>"
  #
  # backend "s3" {
  #   key     = "appther-chatbot/terraform.tfstate"
  #   region  = "us-east-1"
  #   encrypt = true
  # }
}
