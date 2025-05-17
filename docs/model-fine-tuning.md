---
layout: page
title: Model Fine-tuning in AWS Bedrock
---

# Model Fine-tuning in AWS Bedrock

This guide covers the model fine-tuning capabilities in AWS Bedrock, explaining both the concepts and implementation approaches for achieving better performance on specific tasks.

## Introduction to Fine-tuning

Fine-tuning involves taking a pre-trained foundation model and further training it on a specific dataset to enhance its performance for particular use cases. In AWS Bedrock, fine-tuning allows you to:

- Improve model performance on domain-specific tasks
- Optimize for specific response formats or styles
- Reduce hallucinations for enterprise data
- Achieve better performance with fewer examples at inference time

## Supported Models for Fine-tuning

AWS Bedrock supports fine-tuning for select foundation models:

| Foundation Model | Fine-tuning Support | Supported Fine-tuning Methods |
|------------------|---------------------|-------------------------------|
| Anthropic Claude | Yes                 | Instruction fine-tuning       |
| Amazon Titan     | Yes                 | Full fine-tuning, PEFT        |
| Meta Llama 2     | Yes                 | Full fine-tuning, PEFT        |
| Cohere Command   | Yes                 | Instruction fine-tuning       |
| AI21 Jurassic    | No                  | -                             |
| Stability AI     | No                  | -                             |

## Fine-tuning Approaches

### 1. Full Fine-tuning

Full fine-tuning updates all model parameters during the training process.

**Advantages:**
- Maximum performance gain
- Complete model customization

**Disadvantages:**
- Computationally expensive
- Requires substantial training data
- Higher cost
- Increased risk of catastrophic forgetting

### 2. Parameter-Efficient Fine-tuning (PEFT)

PEFT methods update only a small subset of parameters or add trainable adapter layers.

**Advantages:**
- More cost-efficient
- Requires less data
- Faster training
- Reduced risk of catastrophic forgetting

**Disadvantages:**
- Potentially smaller performance gains than full fine-tuning

### 3. Instruction Fine-tuning

Focuses on teaching models to follow specific instructions through examples.

**Advantages:**
- Well-suited for conversational models
- Efficient for teaching response formats
- Requires moderate amounts of data

**Disadvantages:**
- Limited to learning from explicit examples
- May not achieve the same level of domain adaptation as other methods

## Fine-tuning Process in AWS Bedrock

### Prerequisites

1. **AWS Account with Bedrock Access**: Ensure you have access to AWS Bedrock and the necessary permissions.
2. **Training Data**: Prepare your training dataset in the required format.
3. **Amazon S3 Bucket**: Create an S3 bucket to store training data and model artifacts.

### Step 1: Data Preparation

Format your training data according to the model's requirements. For example, with Claude:

```json
{
  "examples": [
    {
      "input": "User query or instruction",
      "output": "Desired model response"
    },
    ...
  ]
}
```

For other models like Titan or Llama 2, different formats may be required.

### Step 2: Upload Training Data to S3

Upload your formatted training data to an S3 bucket:

```bash
aws s3 cp training_data.json s3://your-bucket/fine-tuning-data/ --profile aws --region us-west-2
```

### Step 3: Create a Fine-tuning Job

Using the AWS CLI:

```bash
aws bedrock create-model-customization-job \
  --customization-type FINE_TUNING \
  --base-model-identifier anthropic.claude-v2 \
  --job-name "claude-customer-support-fine-tuning" \
  --training-data-config "{\"s3Uri\": \"s3://your-bucket/fine-tuning-data/training_data.json\"}" \
  --validation-data-config "{\"s3Uri\": \"s3://your-bucket/fine-tuning-data/validation_data.json\"}" \
  --hyper-parameters "{\"epochCount\": 3, \"batchSize\": 8, \"learningRate\": 0.00005}" \
  --output-data-config "{\"s3Uri\": \"s3://your-bucket/fine-tuning-output/\"}" \
  --profile aws \
  --region us-west-2
```

In Python with Boto3:

```python
import boto3
from utils.profile_manager import get_bedrock_client

# Create a Bedrock client
bedrock = get_bedrock_client(profile_name="aws")

response = bedrock.create_model_customization_job(
    customizationType="FINE_TUNING",
    baseModelIdentifier="anthropic.claude-v2",
    jobName="claude-customer-support-fine-tuning",
    trainingDataConfig={
        "s3Uri": "s3://your-bucket/fine-tuning-data/training_data.json"
    },
    validationDataConfig={
        "s3Uri": "s3://your-bucket/fine-tuning-data/validation_data.json"
    },
    hyperParameters={
        "epochCount": 3,
        "batchSize": 8,
        "learningRate": 0.00005
    },
    outputDataConfig={
        "s3Uri": "s3://your-bucket/fine-tuning-output/"
    }
)

job_id = response["jobArn"]
print(f"Created fine-tuning job: {job_id}")
```

### Step 4: Monitor Fine-tuning Progress

```python
def check_fine_tuning_status(job_id):
    """Monitor the status of a fine-tuning job."""
    response = bedrock.get_model_customization_job(
        jobIdentifier=job_id
    )
    
    status = response["status"]
    print(f"Job status: {status}")
    
    if "metrics" in response:
        metrics = response["metrics"]
        print(f"Training loss: {metrics.get('trainingLoss')}")
        print(f"Validation loss: {metrics.get('validationLoss')}")
    
    return status, response
```

### Step 5: Using the Fine-tuned Model

Once fine-tuning completes, you can use your custom model:

```python
# Get the fine-tuned model ID
status, job_details = check_fine_tuning_status(job_id)
if status == "COMPLETED":
    model_id = job_details["outputModelArn"]
    
    # Use the fine-tuned model
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "prompt": "How can I reset my password?",
            "max_tokens_to_sample": 500,
            "temperature": 0.7,
        })
    )
    
    result = json.loads(response["body"].read())
    print(result["completion"])
```

## Best Practices for Fine-tuning

### Data Preparation

1. **Quality over Quantity**: Carefully curate high-quality examples that demonstrate the desired behavior.
2. **Diverse Examples**: Include a diverse set of examples covering different aspects of your use case.
3. **Balance**: Ensure balanced representation of different classes or response types.
4. **Clear Instructions**: For instruction fine-tuning, provide clear and consistent instructions.
5. **Data Validation**: Thoroughly validate your training data before submission.

### Hyperparameter Selection

1. **Learning Rate**: Use a small learning rate (1e-5 to 5e-5) to prevent catastrophic forgetting.
2. **Batch Size**: Adjust according to available resources (typically 4-32).
3. **Epochs**: Start with 2-3 epochs and adjust based on validation performance.
4. **Evaluation Frequency**: Evaluate frequently to catch overfitting early.

### Evaluation

1. **Define Clear Metrics**: Establish clear success metrics aligned with your use case.
2. **Hold-out Test Set**: Always evaluate on a separate test set not used in training.
3. **Human Evaluation**: Complement automated metrics with human evaluation.
4. **Baseline Comparison**: Compare against the base model and other baselines.

## Common Use Cases and Examples

### Use Case: Customer Support Automation

Fine-tune a model to handle customer support queries with company-specific knowledge:

```python
# Example training data format for customer support
training_data = {
    "examples": [
        {
            "input": "How do I change my shipping address for my order #12345?",
            "output": "To change your shipping address for order #12345, please log into your account, go to 'Order History', select the order, and click 'Edit Shipping Details'. If the order has already been processed, please contact customer support at support@example.com for assistance."
        },
        # More examples...
    ]
}
```

### Use Case: Legal Document Analysis

Fine-tune a model to extract specific clauses and information from legal documents:

```python
# Example training data for legal document analysis
training_data = {
    "examples": [
        {
            "input": "AGREEMENT dated October 15, 2023 between ABC Corp ('Provider') and XYZ Inc ('Client'). Term: This agreement shall commence on November 1, 2023 and continue for 24 months thereafter. Termination: Either party may terminate with 30 days written notice.",
            "output": {
                "parties": ["ABC Corp (Provider)", "XYZ Inc (Client)"],
                "effective_date": "November 1, 2023",
                "term_length": "24 months",
                "termination_notice": "30 days"
            }
        },
        # More examples...
    ]
}
```

## Cost Considerations

Fine-tuning introduces additional costs beyond standard inference:

1. **Training Costs**: Based on model size, training data size, and number of epochs
2. **Storage Costs**: S3 storage for training data and model artifacts
3. **Inference Costs**: Fine-tuned models may have different pricing than base models

Always estimate costs before starting a fine-tuning job, especially for large models or datasets.

## Troubleshooting

### Common Issues and Solutions

1. **Training Job Fails**: 
   - Check data format matches model requirements
   - Verify S3 bucket permissions
   - Review training log for specific errors

2. **Poor Performance**:
   - Add more diverse training examples
   - Adjust hyperparameters (try a lower learning rate)
   - Increase training epochs
   - Check for data quality issues

3. **Overfitting**:
   - Reduce number of epochs
   - Add more diverse training examples
   - Implement early stopping

## Conclusion

Fine-tuning foundation models in AWS Bedrock can significantly improve performance for specific use cases. By following the best practices outlined in this guide and iteratively refining your approach, you can create custom models that better meet your specific requirements with fewer examples at inference time.

Remember that fine-tuning is an iterative process, and it often takes several attempts to achieve optimal results. Start small, experiment, and scale up based on results.