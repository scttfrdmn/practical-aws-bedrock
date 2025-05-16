# AWS profile management utilities
# Follows the standards defined in CLAUDE.md

import os
import configparser
from pathlib import Path

def get_profile() -> str:
    """
    Get the AWS profile to use, following these precedence rules:
    1. AWS_PROFILE environment variable if set
    2. 'aws' profile for local testing (default)
    
    Returns:
        str: The AWS profile name to use
    """
    # Check environment variable first
    profile = os.environ.get("AWS_PROFILE")
    
    # If not set, use 'aws' profile as per CLAUDE.md standard
    if not profile:
        profile = "aws"
        
    return profile

def get_region() -> str:
    """
    Get the AWS region to use, following these precedence rules:
    1. AWS_REGION environment variable if set
    2. Region from the active AWS profile configuration
    3. Default to us-west-2 as a fallback
    
    Returns:
        str: The AWS region to use
    """
    # Check environment variable first
    region = os.environ.get("AWS_REGION")
    if region:
        return region
    
    # Get the profile name
    profile_name = get_profile()
    
    # Check the AWS config file
    aws_config_path = os.path.expanduser("~/.aws/config")
    if os.path.exists(aws_config_path):
        config = configparser.ConfigParser()
        config.read(aws_config_path)
        
        # Handle both 'profile aws' and 'default' sections
        profile_section = f"profile {profile_name}"
        
        if profile_section in config:
            if "region" in config[profile_section]:
                return config[profile_section]["region"]
        elif profile_name == "default" and "default" in config:
            if "region" in config["default"]:
                return config["default"]["region"]
    
    # Default to us-west-2 as a fallback (Bedrock is available there)
    return "us-west-2"

def create_session_config() -> dict:
    """
    Create a configuration dictionary for AWS sessions based on 
    the current profile and region settings.
    
    Returns:
        dict: Configuration parameters for boto3.Session
    """
    config = {
        "profile_name": get_profile(),
        "region_name": get_region()
    }
    
    return config

def validate_profile_exists(profile_name: str = None) -> bool:
    """
    Check if a given AWS profile exists in the configuration.
    If no profile name is provided, checks the profile that would be used.
    
    Args:
        profile_name (str, optional): Name of profile to check. Defaults to None.
        
    Returns:
        bool: True if profile exists, False otherwise
    """
    if profile_name is None:
        profile_name = get_profile()
    
    # Check credentials file
    credentials_path = os.path.expanduser("~/.aws/credentials")
    if os.path.exists(credentials_path):
        config = configparser.ConfigParser()
        config.read(credentials_path)
        if profile_name in config:
            return True
    
    # Check config file
    config_path = os.path.expanduser("~/.aws/config")
    if os.path.exists(config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        profile_section = f"profile {profile_name}"
        if profile_section in config:
            return True
        # Check for default
        if profile_name == "default" and "default" in config:
            return True
    
    return False

def print_active_configuration() -> None:
    """
    Print the active AWS configuration information.
    Useful for debugging and confirming settings.
    """
    profile = get_profile()
    region = get_region()
    profile_exists = validate_profile_exists(profile)
    
    print(f"Active AWS Configuration:")
    print(f"  Profile: {profile}{' (not found)' if not profile_exists else ''}")
    print(f"  Region: {region}")
    print(f"  Profile Source: {'Environment variable' if os.environ.get('AWS_PROFILE') else 'Default (aws)'}")
    print(f"  Region Source: {'Environment variable' if os.environ.get('AWS_REGION') else 'AWS config file' if region != 'us-west-2' else 'Default fallback'}")

if __name__ == "__main__":
    # If run directly, print the current configuration
    print_active_configuration()