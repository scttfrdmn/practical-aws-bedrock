"""
Parameter tuning utilities for AWS Bedrock Converse API.

This module provides tools for optimizing and experimenting with different
parameter settings for AWS Bedrock conversational models.
"""

import json
import time
import logging
import uuid
from typing import Dict, Any, Optional, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .simple_chat import ConverseClient


class ParameterTuningClient(ConverseClient):
    """
    A client for parameter tuning with AWS Bedrock Converse API.
    
    This client extends the basic ConverseClient with utilities for
    experimenting with different model parameters and evaluating results.
    """
    
    def __init__(
        self, 
        model_id: str,
        profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        max_retries: int = 3,
        base_backoff: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the parameter tuning client.
        
        Args:
            model_id: The Bedrock model identifier
            profile_name: AWS profile name (defaults to value from get_profile())
            region_name: AWS region name (defaults to value from get_region())
            max_retries: Maximum number of retry attempts for recoverable errors
            base_backoff: Base backoff time (in seconds) for exponential backoff
            logger: Optional logger instance
        """
        super().__init__(
            model_id=model_id,
            profile_name=profile_name,
            region_name=region_name,
            max_retries=max_retries,
            base_backoff=base_backoff,
            logger=logger
        )
    
    def experiment_temperature(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperatures: List[float] = [0.0, 0.3, 0.7, 1.0],
        repeats: int = 1,
        max_tokens: int = 500,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Experiment with different temperature settings.
        
        Args:
            prompt: The prompt to test
            system_prompt: Optional system prompt
            temperatures: List of temperature values to test
            repeats: Number of repeats for each temperature
            max_tokens: Maximum tokens to generate
            other_params: Additional model-specific parameters
            
        Returns:
            Dictionary with experiment results
        """
        results = {}
        
        for temp in temperatures:
            temp_results = []
            
            for i in range(repeats):
                # Create a new conversation for each experiment
                conversation_id = self.create_conversation(system_prompt=system_prompt)
                
                # Send prompt with this temperature
                response = self.send_message(
                    conversation_id=conversation_id,
                    message=prompt,
                    max_tokens=max_tokens,
                    temperature=temp,
                    other_params=other_params
                )
                
                # Get response length metrics
                response_length = len(response)
                response_word_count = len(response.split())
                
                # Store result
                temp_results.append({
                    "response": response,
                    "length_chars": response_length,
                    "word_count": response_word_count,
                })
                
                # Clean up conversation
                self.delete_conversation(conversation_id)
            
            # Aggregate results for this temperature
            results[str(temp)] = {
                "responses": temp_results,
                "average_length": sum(r["length_chars"] for r in temp_results) / len(temp_results),
                "average_word_count": sum(r["word_count"] for r in temp_results) / len(temp_results),
            }
        
        return results
    
    def experiment_system_prompts(
        self, 
        prompt: str,
        system_prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 500,
        other_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Experiment with different system prompts.
        
        Args:
            prompt: The prompt to test
            system_prompts: List of system prompts to test
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            other_params: Additional model-specific parameters
            
        Returns:
            Dictionary with experiment results
        """
        results = {}
        
        for i, system_prompt in enumerate(system_prompts):
            # Create a new conversation with this system prompt
            conversation_id = self.create_conversation(system_prompt=system_prompt)
            
            # Send prompt
            response = self.send_message(
                conversation_id=conversation_id,
                message=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                other_params=other_params
            )
            
            # Store result
            results[f"system_prompt_{i}"] = {
                "system_prompt": system_prompt,
                "response": response,
                "length_chars": len(response),
                "word_count": len(response.split())
            }
            
            # Clean up conversation
            self.delete_conversation(conversation_id)
        
        return results
    
    def parallel_parameter_sweep(
        self, 
        prompt: str,
        parameter_combinations: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        max_workers: int = 5
    ) -> Dict[str, Any]:
        """
        Run parallel experiments with different parameter combinations.
        
        Args:
            prompt: The prompt to test
            parameter_combinations: List of parameter dictionaries to test
            system_prompt: Optional default system prompt
            max_tokens: Default maximum tokens to generate
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary with experiment results
        """
        results = {}
        
        def run_experiment(params: Dict[str, Any], exp_id: int) -> Tuple[int, Dict[str, Any]]:
            """Run a single experiment with the given parameters"""
            # Extract parameters or use defaults
            sys_prompt = params.get("system_prompt", system_prompt)
            temp = params.get("temperature", 0.7)
            tokens = params.get("max_tokens", max_tokens)
            other = params.get("other_params", {})
            
            # Create a new conversation
            conversation_id = self.create_conversation(system_prompt=sys_prompt)
            
            # Get start time
            start_time = time.time()
            
            # Send prompt with parameters
            response = self.send_message(
                conversation_id=conversation_id,
                message=prompt,
                max_tokens=tokens,
                temperature=temp,
                other_params=other
            )
            
            # Calculate time taken
            duration = time.time() - start_time
            
            # Get conversation history for token metrics
            history = self.get_conversation_history(conversation_id)
            
            # Clean up conversation
            self.delete_conversation(conversation_id)
            
            # Return experiment results
            return exp_id, {
                "parameters": params,
                "response": response,
                "duration_seconds": duration,
                "length_chars": len(response),
                "word_count": len(response.split()),
                "message_count": len(history["messages"])
            }
        
        # Run experiments in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_id = {
                executor.submit(run_experiment, params, i): i 
                for i, params in enumerate(parameter_combinations)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_id):
                exp_id, exp_result = future.result()
                results[f"experiment_{exp_id}"] = exp_result
        
        return results
    
    def evaluate_response_quality(
        self, 
        prompt: str,
        reference_answer: str,
        parameter_combinations: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = 500,
        evaluation_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate response quality across different parameter settings.
        
        This method uses a separate evaluation prompt to assess response quality
        based on a reference answer and specified criteria.
        
        Args:
            prompt: The prompt to test
            reference_answer: Reference/ideal answer to compare with
            parameter_combinations: List of parameter dictionaries to test
            system_prompt: Optional default system prompt
            max_tokens: Default maximum tokens to generate
            evaluation_criteria: List of criteria to evaluate (defaults to accuracy/relevance)
            
        Returns:
            Dictionary with evaluation results
        """
        if evaluation_criteria is None:
            evaluation_criteria = [
                "Accuracy - How factually accurate is the response?",
                "Relevance - How relevant is the response to the prompt?",
                "Completeness - How thoroughly does the response address the prompt?",
                "Clarity - How clear and understandable is the response?"
            ]
        
        # Run experiments to get responses
        experiment_results = self.parallel_parameter_sweep(
            prompt=prompt,
            parameter_combinations=parameter_combinations,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        
        # Create an evaluation client
        evaluation_client = ConverseClient(
            model_id=self.model_id,
            profile_name=self.profile_name,
            region_name=self.region_name
        )
        
        # Define evaluation system prompt
        eval_system_prompt = """
        You are a helpful evaluation assistant. Your task is to objectively evaluate the quality
        of a model's response compared to a reference answer. Rate each criterion on a scale of 1-10,
        where 1 is the lowest quality and 10 is the highest quality. Provide a brief explanation
        for each rating. Be fair and consistent in your evaluations.
        """
        
        # Create formatted criteria string
        criteria_str = "\n".join([f"- {criterion}" for criterion in evaluation_criteria])
        
        # Evaluate each experiment
        for exp_id, exp_result in experiment_results.items():
            response = exp_result["response"]
            parameters = exp_result["parameters"]
            
            # Create evaluation prompt
            eval_prompt = f"""
            Please evaluate the following model response based on the provided criteria.
            
            Original Prompt: {prompt}
            
            Reference Answer:
            ---
            {reference_answer}
            ---
            
            Model Response:
            ---
            {response}
            ---
            
            Evaluation Criteria:
            {criteria_str}
            
            For each criterion, provide:
            1. A score from 1-10
            2. A brief explanation for the score
            
            End with an overall score (average) and summary assessment.
            """
            
            # Create conversation for evaluation
            eval_conversation_id = evaluation_client.create_conversation(system_prompt=eval_system_prompt)
            
            # Get evaluation
            evaluation = evaluation_client.send_message(
                conversation_id=eval_conversation_id,
                message=eval_prompt,
                max_tokens=1000,
                temperature=0.3  # Low temperature for consistent evaluations
            )
            
            # Add evaluation to results
            experiment_results[exp_id]["evaluation"] = evaluation
            
            # Clean up evaluation conversation
            evaluation_client.delete_conversation(eval_conversation_id)
        
        return experiment_results


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a parameter tuning client for Claude
    client = ParameterTuningClient(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
    
    try:
        # Example 1: Temperature experiment
        print("\n--- Example 1: Temperature Experiment ---")
        
        temp_results = client.experiment_temperature(
            prompt="Explain quantum computing in simple terms.",
            temperatures=[0.0, 0.5, 1.0],
            repeats=2,
            max_tokens=200
        )
        
        # Print results summary
        print("Temperature experiment results:")
        for temp, results in temp_results.items():
            print(f"\nTemperature: {temp}")
            print(f"Average response length: {results['average_length']:.1f} characters")
            print(f"Average word count: {results['average_word_count']:.1f} words")
            print(f"Sample response: {results['responses'][0]['response'][:100]}...")
        
        # Example 2: System prompt experiment
        print("\n--- Example 2: System Prompt Experiment ---")
        
        system_prompts = [
            "You are a helpful assistant that explains complex topics in simple terms.",
            "You are a scientific expert who provides detailed technical explanations.",
            "You are a teacher helping a 10-year-old understand difficult concepts."
        ]
        
        sys_prompt_results = client.experiment_system_prompts(
            prompt="Explain how a vaccine works.",
            system_prompts=system_prompts,
            max_tokens=300
        )
        
        # Print results summary
        print("System prompt experiment results:")
        for key, result in sys_prompt_results.items():
            print(f"\nSystem prompt: {result['system_prompt'][:50]}...")
            print(f"Response length: {result['length_chars']} characters")
            print(f"Word count: {result['word_count']} words")
            print(f"Sample response: {result['response'][:100]}...")
        
        # Example 3: Parameter sweep
        print("\n--- Example 3: Parameter Sweep ---")
        
        parameter_combinations = [
            {"temperature": 0.0, "max_tokens": 200},
            {"temperature": 0.7, "max_tokens": 200},
            {"temperature": 1.0, "max_tokens": 200},
            {"temperature": 0.7, "max_tokens": 400},
            {"system_prompt": "Be extremely concise.", "temperature": 0.7, "max_tokens": 200},
        ]
        
        sweep_results = client.parallel_parameter_sweep(
            prompt="Explain three key benefits of cloud computing.",
            parameter_combinations=parameter_combinations,
            max_workers=3
        )
        
        # Print results summary
        print("Parameter sweep results:")
        for exp_id, result in sweep_results.items():
            params = result["parameters"]
            print(f"\n{exp_id}:")
            print(f"Parameters: temp={params.get('temperature', 0.7)}, tokens={params.get('max_tokens', 500)}")
            if "system_prompt" in params:
                print(f"System prompt: {params['system_prompt']}")
            print(f"Response time: {result['duration_seconds']:.2f} seconds")
            print(f"Response length: {result['length_chars']} characters")
            print(f"Word count: {result['word_count']} words")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Get metrics
    metrics = client.get_metrics()
    print("\nClient Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")