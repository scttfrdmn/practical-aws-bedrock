# Utilities for monitoring and managing AWS Bedrock quota usage

import boto3
import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional
from utils.profile_manager import get_profile
from utils.visualization_config import SVG_CONFIG

class QuotaMonitor:
    """
    A class to monitor AWS Bedrock quota usage and provide optimization recommendations.
    Follows the AWS profile guidelines in CLAUDE.md.
    """
    
    def __init__(self, profile_name=None):
        """
        Initialize the QuotaMonitor with AWS clients.
        
        Args:
            profile_name: AWS profile to use. Defaults to get_profile() value.
        """
        # Use the configured profile (defaults to 'aws' for local testing)
        self.profile_name = profile_name or get_profile()
        self.session = boto3.Session(profile_name=self.profile_name)
        self.cloudwatch = self.session.client('cloudwatch')
        self.service_quotas = self.session.client('service-quotas')
        self.bedrock = self.session.client('bedrock')
        self.bedrock_runtime = self.session.client('bedrock-runtime')
    
    def get_bedrock_quotas(self) -> List[Dict]:
        """
        Retrieve all AWS Bedrock quotas for the current account.
        
        Returns:
            List of quota dictionaries
        """
        response = self.service_quotas.list_service_quotas(ServiceCode='bedrock')
        return response['Quotas']
    
    def get_model_specific_quotas(self, model_id: str) -> List[Dict]:
        """
        Find quotas specific to a model ID.
        
        Args:
            model_id: The model identifier (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            
        Returns:
            List of quota dictionaries relevant to the model
        """
        # Get all quotas
        all_quotas = self.get_bedrock_quotas()
        
        # Extract model family from model_id
        model_family = model_id.split('.')[0] if '.' in model_id else model_id
        
        # Filter quotas by model family
        return [
            quota for quota in all_quotas 
            if model_family.lower() in quota['QuotaName'].lower()
        ]
    
    def get_usage_metrics(
        self, 
        model_id: str, 
        hours: int = 24, 
        metric_names: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get usage metrics for a specific model from CloudWatch.
        
        Args:
            model_id: The model identifier
            hours: Number of hours to look back
            metric_names: List of metrics to retrieve (default: all relevant metrics)
            
        Returns:
            Dictionary mapping metric names to their datapoints
        """
        if metric_names is None:
            metric_names = [
                'InvokeModel', 
                'InvokeModelClientErrors',
                'InvokeModelUserErrors',
                'InvokeModelThrottled'
            ]
        
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(hours=hours)
        results = {}
        
        for metric_name in metric_names:
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Bedrock',
                MetricName=metric_name,
                Dimensions=[
                    {'Name': 'ModelId', 'Value': model_id}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour periods
                Statistics=['Sum']
            )
            
            results[metric_name] = response['Datapoints']
        
        return results
    
    def check_throttle_rate(self, model_id: str, hours: int = 24) -> float:
        """
        Calculate the throttle rate for a model.
        
        Args:
            model_id: The model identifier
            hours: Number of hours to look back
            
        Returns:
            Percentage of requests that were throttled
        """
        metrics = self.get_usage_metrics(
            model_id, 
            hours, 
            ['InvokeModel', 'InvokeModelThrottled']
        )
        
        total_invocations = sum(point['Sum'] for point in metrics['InvokeModel'])
        total_throttled = sum(point['Sum'] for point in metrics['InvokeModelThrottled'])
        
        if total_invocations + total_throttled == 0:
            return 0.0
        
        return (total_throttled / (total_invocations + total_throttled)) * 100
    
    def get_model_throughput_capacity(self, model_id: str) -> Dict:
        """
        Get the provisioned throughput capacity for a specific model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dictionary with capacity information
        """
        try:
            response = self.bedrock.get_foundation_model_throughput_capacity(
                modelId=model_id
            )
            return response
        except self.bedrock.exceptions.ResourceNotFoundException:
            # Handle case where model doesn't have provisioned throughput
            return {"status": "Not available", "modelId": model_id}
    
    def generate_quota_usage_report(self, model_id: str, hours: int = 24) -> Dict:
        """
        Generate a comprehensive report of quota usage for a model.
        
        Args:
            model_id: The model identifier
            hours: Number of hours to look back
            
        Returns:
            Dictionary with usage statistics and recommendations
        """
        # Get quota information
        quotas = self.get_model_specific_quotas(model_id)
        
        # Get usage metrics
        metrics = self.get_usage_metrics(model_id, hours)
        
        # Calculate throttle rate
        throttle_rate = self.check_throttle_rate(model_id, hours)
        
        # Get throughput capacity
        throughput = self.get_model_throughput_capacity(model_id)
        
        # Generate recommendations based on usage patterns
        recommendations = []
        
        if throttle_rate > 10:
            recommendations.append(
                "High throttle rate detected. Consider requesting a quota increase."
            )
        
        if throttle_rate > 0:
            recommendations.append(
                "Implement exponential backoff with jitter for retrying throttled requests."
            )
        
        # Regular, periodic patterns might benefit from scheduling
        if len(metrics['InvokeModel']) > 10:
            peak_usage = max([point['Sum'] for point in metrics['InvokeModel']], default=0)
            avg_usage = sum([point['Sum'] for point in metrics['InvokeModel']]) / len(metrics['InvokeModel'])
            
            if peak_usage > avg_usage * 2:
                recommendations.append(
                    "Usage shows peaks and valleys. Consider request scheduling to distribute load evenly."
                )
        
        # Return comprehensive report
        return {
            "model_id": model_id,
            "quotas": quotas,
            "metrics": metrics,
            "throttle_rate": throttle_rate,
            "throughput_capacity": throughput,
            "recommendations": recommendations,
            "report_period_hours": hours,
            "report_generated": datetime.datetime.utcnow().isoformat()
        }
    
    def visualize_quota_usage(
        self, 
        model_id: str, 
        hours: int = 24, 
        output_file: str = "quota_usage.svg"
    ) -> Optional[str]:
        """
        Create an SVG visualization of quota usage for a model.
        
        Args:
            model_id: The model identifier
            hours: Number of hours to look back
            output_file: Path to save the SVG file
            
        Returns:
            Path to the saved SVG file, or None if no data
        """
        # Get usage metrics
        metrics = self.get_usage_metrics(
            model_id, 
            hours, 
            ['InvokeModel', 'InvokeModelThrottled']
        )
        
        # Process data for visualization
        data = []
        
        for point in metrics['InvokeModel']:
            data.append({
                'timestamp': point['Timestamp'],
                'invocations': point['Sum'],
                'type': 'Successful'
            })
        
        for point in metrics['InvokeModelThrottled']:
            data.append({
                'timestamp': point['Timestamp'],
                'invocations': point['Sum'],
                'type': 'Throttled'
            })
        
        df = pd.DataFrame(data)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        if not df.empty:
            # Pivot data for stacked bar chart
            pivot_df = df.pivot_table(
                index='timestamp', 
                columns='type', 
                values='invocations',
                fill_value=0
            )
            
            # Sort by timestamp
            pivot_df = pivot_df.sort_index()
            
            # Create stacked bar chart
            pivot_df.plot(
                kind='bar', 
                stacked=True, 
                ax=plt.gca(),
                color=['#2ca02c', '#d62728']  # Green for success, red for throttled
            )
            
            plt.title(f'API Usage for {model_id}', fontsize=16)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Request Count', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save as SVG
            plt.savefig(output_file, **SVG_CONFIG)
            plt.close()
            
            return output_file
        else:
            print("No data available for the specified time period")
            return None
    
    def simulate_quota_limits(
        self, 
        model_id: str,
        requests_per_minute: int,
        tokens_per_request: int,
        duration_minutes: int = 10
    ) -> Dict:
        """
        Simulate how a workload would perform under current quota limits.
        
        Args:
            model_id: The model identifier
            requests_per_minute: Desired RPM
            tokens_per_request: Average tokens per request (input + output)
            duration_minutes: Duration of simulation in minutes
            
        Returns:
            Dictionary with simulation results
        """
        # Get relevant quotas
        quotas = self.get_model_specific_quotas(model_id)
        
        # Extract RPM and TPM limits if available
        rpm_limit = None
        tpm_limit = None
        
        for quota in quotas:
            if "Requests per minute" in quota['QuotaName']:
                rpm_limit = quota['Value']
            elif "Tokens per minute" in quota['QuotaName']:
                tpm_limit = quota['Value']
        
        # Use default conservative values if not found
        rpm_limit = rpm_limit or 100
        tpm_limit = tpm_limit or 10000
        
        # Calculate expected throughput
        expected_rpm = min(requests_per_minute, rpm_limit)
        expected_tpm = min(requests_per_minute * tokens_per_request, tpm_limit)
        
        # Calculate throttling
        rpm_throttle_rate = max(0, (requests_per_minute - rpm_limit) / requests_per_minute * 100) if requests_per_minute > 0 else 0
        tpm_throttle_rate = max(0, (requests_per_minute * tokens_per_request - tpm_limit) / (requests_per_minute * tokens_per_request) * 100) if requests_per_minute > 0 else 0
        
        # Overall throttle rate is the maximum of the two
        overall_throttle_rate = max(rpm_throttle_rate, tpm_throttle_rate)
        
        # Calculate expected successful requests
        expected_successful_requests = int(requests_per_minute * duration_minutes * (1 - overall_throttle_rate/100))
        
        # Return simulation results
        return {
            "model_id": model_id,
            "input_parameters": {
                "requests_per_minute": requests_per_minute,
                "tokens_per_request": tokens_per_request,
                "duration_minutes": duration_minutes
            },
            "quota_limits": {
                "rpm_limit": rpm_limit,
                "tpm_limit": tpm_limit
            },
            "expected_throughput": {
                "effective_rpm": expected_rpm,
                "effective_tpm": expected_tpm
            },
            "throttling": {
                "rpm_throttle_rate": rpm_throttle_rate,
                "tpm_throttle_rate": tpm_throttle_rate,
                "overall_throttle_rate": overall_throttle_rate
            },
            "expected_successful_requests": expected_successful_requests,
            "limiting_factor": "RPM" if rpm_throttle_rate > tpm_throttle_rate else "TPM"
        }
    
    def export_report_to_json(self, report: Dict, filename: str) -> None:
        """
        Export a quota usage report to a JSON file.
        
        Args:
            report: The report dictionary
            filename: Path to save the JSON file
        """
        # Convert datetime objects to strings for JSON serialization
        def json_serial(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filename, 'w') as f:
            json.dump(report, f, default=json_serial, indent=2)

# Example usage
if __name__ == "__main__":
    monitor = QuotaMonitor()
    
    # Example model ID
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    # Generate and export a report
    report = monitor.generate_quota_usage_report(model_id)
    monitor.export_report_to_json(report, "quota_report.json")
    
    # Visualize usage
    monitor.visualize_quota_usage(model_id, output_file="quota_usage.svg")
    
    # Simulate a workload
    simulation = monitor.simulate_quota_limits(
        model_id=model_id,
        requests_per_minute=100,
        tokens_per_request=1000,
        duration_minutes=60
    )
    print(json.dumps(simulation, indent=2))