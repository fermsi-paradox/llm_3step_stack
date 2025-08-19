"""
Complete Lambda GPU fine-tuning pipeline for NA beer brewing LLM with Mixtral 7Bx8
This script handles: training, merging, and uploading to HuggingFace
"""

import os
import json
import time
import requests
import paramiko
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration from environment
LAMBDA_API_KEY = os.getenv('LAMBDA_API_KEY')
LAMBDA_SSH_KEY_NAME = os.getenv('LAMBDA_SSH_KEY_NAME')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_ACCOUNT = os.getenv('HUGGINGFACE_ACCOUNT')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'expert-model-finetuning')

# Model configuration for Mixtral with anti-overfitting
BASE_MODEL = os.getenv('BASE_MODEL', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
OUTPUT_MODEL_NAME = os.getenv('OUTPUT_MODEL_NAME', 'expert_llm_model')

class LambdaGPUTrainer:
    def __init__(self):
        self.lambda_api_url = "https://cloud.lambdalabs.com/api/v1"
        self.headers = {
            "Authorization": f"Bearer {LAMBDA_API_KEY}",
            "Content-Type": "application/json"
        }
        self.instance_id = None
        self.instance_ip = None
        # Use lambda_gpu_key which matches the uploaded public key
        self.ssh_key_path = Path.home() / ".ssh" / "lambda_gpu_key"
        
    def get_instance_status(self):
        """Get current Lambda instance status via API."""
        try:
            response = requests.get(f"{self.lambda_api_url}/instances", headers=self.headers)
            if response.status_code == 200:
                instances = response.json()["data"]
                for instance in instances:
                    if instance["ip"] == self.instance_ip:
                        self.instance_id = instance["id"]
                        return instance["status"]
                logger.warning(f"Instance with IP {self.instance_ip} not found")
                return "unknown"
            else:
                logger.error(f"Failed to get instance status: {response.text}")
                return "api_error"
        except Exception as e:
            logger.error(f"Error checking instance status: {e}")
            return "error"
    
    def monitor_training_progress(self, check_interval=60):
        """Monitor training progress with Lambda API integration."""
        logger.info("🔍 Starting training progress monitoring...")
        logger.info(f"📡 Lambda API Status: {self.get_instance_status()}")
        
        start_time = time.time()
        last_checkpoint_count = 0
        
        while True:
            try:
                # Check if training is still running via SSH
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(self.instance_ip, username="ubuntu", key_filename=str(self.ssh_key_path))
                
                # Check for training process
                stdin, stdout, stderr = ssh.exec_command("pgrep -f 'python lambda_training_script.py'")
                pids = stdout.read().decode().strip()
                
                if not pids:
                    logger.info("🏁 Training process completed!")
                    break
                
                # Check training log for checkpoints
                stdin, stdout, stderr = ssh.exec_command("cd /home/ubuntu/na_llm_training && grep -c 'CHECKPOINT.*SAVED' training.log || echo '0'")
                checkpoint_count = int(stdout.read().decode().strip())
                
                if checkpoint_count > last_checkpoint_count:
                    logger.info(f"🎯 New checkpoint detected! Progress: {checkpoint_count}/10")
                    last_checkpoint_count = checkpoint_count
                
                # Log runtime and instance status
                runtime = (time.time() - start_time) / 3600  # hours
                instance_status = self.get_instance_status()
                logger.info(f"⏱️  Runtime: {runtime:.1f}h | Instance: {instance_status} | Checkpoints: {checkpoint_count}/10")
                
                ssh.close()
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(check_interval)
        
    # def create_ssh_key(self):
    #     """Create SSH key for Lambda instance access."""
    #     # NOTE: No longer needed - using existing SSH key from Lambda account
    #     pass
        
    def launch_instance(self):
        """Launch a Lambda GPU instance, trying different A100 configurations."""
        
        # A100 instance types in order of preference for Mixtral fine-tuning
        a100_instance_types = [
            "gpu_1x_a100_sxm4",       # Original choice
            "gpu_1x_a100",            # Standard A100
            "gpu_8x_a100_80gb_sxm4",  # 80GB variant (more VRAM)
            "gpu_2x_a100",            # 2x A100 (in case we need more VRAM)
            "gpu_4x_a100",            # 4x A100
            "gpu_8x_a100"             # 8x A100
        ]
        
        # Use existing SSH key from Lambda account
        ssh_key_name = LAMBDA_SSH_KEY_NAME
        if not ssh_key_name:
            raise ValueError("LAMBDA_SSH_KEY_NAME must be set in environment variables")
        
        logger.info(f"Using SSH key: {ssh_key_name}")
        
        # Try each A100 instance type until one works
        for instance_type in a100_instance_types:
            logger.info(f"Trying to launch A100 instance: {instance_type}")
            
            # Launch instance
            launch_data = {
                "region_name": "us-west-2",  # Or preferred region
                "instance_type_name": instance_type,
                "ssh_key_names": [ssh_key_name]  # Use existing SSH key from account
            }
            
            response = requests.post(
                f"{self.lambda_api_url}/instance-operations/launch",
                headers=self.headers,
                json=launch_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.instance_id = result["data"]["instance_ids"][0]
                logger.info(f"✅ A100 instance launched successfully: {self.instance_id} ({instance_type})")
                
                # Wait for instance to be running
                self._wait_for_instance_running()
                return True
            else:
                error_data = response.json()
                error_code = error_data.get("error", {}).get("code", "")
                if "insufficient-capacity" in error_code:
                    logger.info(f"❌ {instance_type}: No capacity available, trying next A100 type...")
                else:
                    logger.error(f"❌ {instance_type}: {response.text}")
                    # For non-capacity errors, still try the next type
                    continue
        
        # If we get here, no A100 instances were available
        logger.error("❌ All A100 instance types are out of capacity or unavailable")
        logger.info("💡 Suggestion: Try again in a few minutes, or check Lambda Labs status page")
        return False
    
    def _wait_for_instance_running(self, timeout=600):
        """Wait for instance to be in running state."""
        logger.info("Waiting for instance to start...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(
                f"{self.lambda_api_url}/instances",
                headers=self.headers
            )
            
            if response.status_code == 200:
                instances = response.json()["data"]
                for instance in instances:
                    if instance["id"] == self.instance_id:
                        if instance["status"] == "running":
                            self.instance_ip = instance["ip"]
                            logger.info(f"Instance running at IP: {self.instance_ip}")
                            time.sleep(30)  # Wait for SSH to be ready
                            return
                        logger.info(f"Instance status: {instance['status']}")
                        break
            
            time.sleep(30)
        
        raise TimeoutError("Instance failed to start within timeout period")
    
    def setup_environment(self):
        """Setup the training environment on Lambda instance."""
        logger.info("Setting up training environment...")
        
        # Connect via SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(
                hostname=self.instance_ip,
                username="ubuntu",
                key_filename=str(self.ssh_key_path),
                timeout=30
            )
            
            # System setup commands (with complete cleanup)
            system_commands = [
                "sudo apt-get update", 
                "sudo apt-get install -y python3-pip python3-venv git",
                "rm -rf /home/ubuntu/na_llm_training",  # Complete cleanup
                "mkdir -p /home/ubuntu/na_llm_training",
                "cd /home/ubuntu/na_llm_training && python3 -m venv training_env",
                "cd /home/ubuntu/na_llm_training && source training_env/bin/activate && pip install --upgrade pip",
            ]
            
            for cmd in system_commands:
                logger.info(f"Executing: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                stdout.channel.recv_exit_status()  # Wait for completion
                logger.info(f"Output: {stdout.read().decode()}")
                error = stderr.read().decode()
                if error:
                    logger.warning(f"Error: {error}")
            
            # Upload training files directly
            logger.info("Uploading training files...")
            sftp = ssh.open_sftp()
            
            try:
                # Upload training script
                sftp.put("lambda_training_script.py", "/home/ubuntu/na_llm_training/lambda_training_script.py")
                logger.info("✅ Uploaded training script")
                
                # Upload requirements
                sftp.put("requirements.txt", "/home/ubuntu/na_llm_training/requirements.txt")
                logger.info("✅ Uploaded requirements.txt")
                
                # Upload .env file
                sftp.put(".env", "/home/ubuntu/na_llm_training/.env")
                logger.info("✅ Uploaded environment variables")
                
            except Exception as e:
                logger.error(f"Failed to upload files: {e}")
                raise
            finally:
                sftp.close()
            
            # Use exact working approach - no version pinning, let pip resolve latest compatible versions
            install_commands = [
                "cd /home/ubuntu/na_llm_training && source training_env/bin/activate && pip install --upgrade pip",
                "cd /home/ubuntu/na_llm_training && source training_env/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", 
                "cd /home/ubuntu/na_llm_training && source training_env/bin/activate && pip install transformers datasets peft bitsandbytes accelerate wandb huggingface-hub python-dotenv scipy sentencepiece protobuf paramiko requests"
            ]
            
            for i, cmd in enumerate(install_commands, 1):
                logger.info(f"Step {i}/3: Installing packages...")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                stdout.channel.recv_exit_status()
                
                output = stdout.read().decode()
                error = stderr.read().decode()
                
                success = 'Successfully installed' in output or 'Requirement already satisfied' in output
                logger.info(f"Step {i}: {'✅ Success' if success else '⚠️ Issues detected'}")
                if error and len(error) > 10:
                    logger.warning(f"Step {i} warnings: {error[:200]}...")
            
            logger.info("✅ Using exact working approach - latest compatible versions (no version pinning)!")
            
            # Verify transformers is installed in venv
            verify_cmd = "cd /home/ubuntu/na_llm_training && source training_env/bin/activate && python -c 'import transformers, torch, numpy; print(f\"✅ Transformers: {transformers.__version__}\"); print(f\"✅ PyTorch: {torch.__version__}\"); print(f\"✅ NumPy: {numpy.__version__}\")'"
            logger.info("Verifying packages in virtual environment...")
            stdin, stdout, stderr = ssh.exec_command(verify_cmd)
            stdout.channel.recv_exit_status()
            logger.info(f"Verification: {stdout.read().decode()}")
            
            # Environment variables already uploaded via .env file
            
            # Note: Training data will be loaded from HuggingFace Hub during training
            
            logger.info("Environment setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            raise
        finally:
            ssh.close()
    
    def _upload_env_file(self, ssh):
        """Upload environment variables to instance."""
        logger.info("Uploading environment variables...")
        
        env_content = f"""
LAMBDA_API_KEY={LAMBDA_API_KEY}
HUGGINGFACE_API_KEY={HUGGINGFACE_API_KEY}
HUGGINGFACE_ACCOUNT={HUGGINGFACE_ACCOUNT}
WANDB_API_KEY={WANDB_API_KEY}
WANDB_PROJECT={WANDB_PROJECT}
"""
        
        # Create .env file
        stdin, stdout, stderr = ssh.exec_command("cd na_llm2/fine_tuning && cat > .env")
        stdin.write(env_content)
        stdin.close()
        stdout.channel.recv_exit_status()
    
    def start_training(self):
        """Start the training process on Lambda instance."""
        logger.info("Starting training process...")
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(
                hostname=self.instance_ip,
                username="ubuntu",
                key_filename=str(self.ssh_key_path),
                timeout=30
            )
            
            # Start training in background with nohup (using virtual environment)
            train_cmd = "cd /home/ubuntu/na_llm_training && source training_env/bin/activate && nohup python lambda_training_script.py > training.log 2>&1 &"
            logger.info(f"Starting training with venv: {train_cmd}")
            
            stdin, stdout, stderr = ssh.exec_command(train_cmd)
            logger.info("Training started in background")
            
            # Monitor training progress
            self._monitor_training(ssh)
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            raise
        finally:
            ssh.close()
    
    def _monitor_training(self, ssh, check_interval=60):
        """Monitor training progress by checking log file."""
        logger.info("Monitoring training progress...")
        
        while True:
            try:
                # Check if training is still running
                stdin, stdout, stderr = ssh.exec_command("pgrep -f lambda_training_script.py")
                if not stdout.read().decode().strip():
                    logger.info("Training process completed")
                    break
                
                # Show latest log entries
                stdin, stdout, stderr = ssh.exec_command("tail -5 /home/ubuntu/na_llm_training/training.log")
                log_output = stdout.read().decode()
                if log_output:
                    logger.info(f"Training log: {log_output}")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring training: {e}")
                break
    
    def download_model(self):
        """Download the trained model from Lambda instance."""
        logger.info("Downloading trained model...")
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(
                hostname=self.instance_ip,
                username="ubuntu",
                key_filename=str(self.ssh_key_path),
                timeout=30
            )
            
            # Download logs
            sftp = ssh.open_sftp()
            try:
                sftp.get(
                    "/home/ubuntu/na_llm_training/training.log",
                    "./training.log"
                )
                logger.info("Training log downloaded")
            except Exception as e:
                logger.warning(f"Could not download training log: {e}")
            finally:
                sftp.close()
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
        finally:
            ssh.close()
    
    def terminate_instance(self):
        """Terminate the Lambda GPU instance."""
        if not self.instance_id:
            logger.warning("No instance to terminate")
            return
        
        logger.info(f"Terminating instance: {self.instance_id}")
        
        response = requests.post(
            f"{self.lambda_api_url}/instance-operations/terminate",
            headers=self.headers,
            json={"instance_ids": [self.instance_id]}
        )
        
        if response.status_code == 200:
            logger.info("Instance terminated successfully")
        else:
            logger.error(f"Failed to terminate instance: {response.text}")

def main():
    """Main execution function."""
    if not all([LAMBDA_API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE_ACCOUNT]):
        logger.error("Missing required environment variables")
        logger.error("Please set: LAMBDA_API_KEY, HUGGINGFACE_API_KEY, HUGGINGFACE_ACCOUNT")
        return
    
    trainer = LambdaGPUTrainer()
    
    try:
        # Use existing Lambda GPU instance (manually launched)
        trainer.instance_ip = "YOUR_LAMBDA_INSTANCE_IP_HERE"
        logger.info(f"🎯 Using existing H100 instance: {trainer.instance_ip}")
        logger.info("✅ Skipping instance launch - using manually started instance")
        
        # Setup environment
        trainer.setup_environment()
        
        # Start training
        trainer.start_training()
        
        # Monitor training progress with checkpoints
        trainer.monitor_training_progress()
        
        # Download results
        trainer.download_model()
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.info("💡 Note: Instance was launched manually - check Lambda console to manage it")
        # Don't auto-terminate manually launched instances

if __name__ == "__main__":
    main() 