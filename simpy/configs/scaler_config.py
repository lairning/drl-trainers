azure_config_str = '''# An unique identifier for the head node and workers of this cluster.
cluster_name: {}

# The maximum number of workers nodes to launch in addition to the head
# node. This takes precedence over min_workers. min_workers default to 0.
min_workers: 0

max_workers: 2

# The autoscaler will scale up the cluster faster with higher upscaling speed.
# E.g., if the task requires adding more nodes then autoscaler will gradually
# scale up the cluster in chunks of upscaling_speed*currently_running_nodes.
# This number should be > 0.
upscaling_speed: 1.0

# If a node is idle for this many minutes, it will be removed.
idle_timeout_minutes: 5

# Cloud-provider specific configuration.
provider:
    type: azure
    location: westeurope
    resource_group: lairning

# How Ray will authenticate with newly launched nodes.
auth:
    ssh_user: ubuntu
    # you must specify paths to matching private and public key pair files
    # use `ssh-keygen -t rsa -b 4096` to generate a new ssh key pair
    ssh_private_key: ~/.ssh/id_rsa
    # changes to this should match what is specified in file_mounts
    ssh_public_key: ~/.ssh/id_rsa.pub

# Provider-specific config for the head node, e.g. instance type.
head_node:
    azure_arm_parameters:
        # Changed to B1S that are free with the Azure Free Subscritpion
        vmSize: Standard_D4s_v3
        # List images https://docs.microsoft.com/en-us/azure/virtual-machines/linux/cli-ps-findimage
        imagePublisher: microsoft-dsvm
        imageOffer: ubuntu-1804
        imageSku: 1804-gen2
        imageVersion: 21.01.21

# Provider-specific config for worker nodes, e.g. instance type.
worker_nodes:
    azure_arm_parameters:
        # Changed to B1S that are free with the Azure Free Subscritpion
        vmSize: Standard_D2s_v3
        # List images https://docs.microsoft.com/en-us/azure/virtual-machines/linux/cli-ps-findimage
        imagePublisher: microsoft-dsvm
        imageOffer: ubuntu-1804
        imageSku: 1804-gen2
        imageVersion: 21.01.21
        # optionally set priority to use Spot instances
        # priority: Spot

file_mounts: {{
    '/home/ubuntu/trainer': '~/drl-trainers/simpy/{}',
}}

# List of shell commands to run to set up nodes.
setup_commands:
    - echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
    - echo 'conda activate py37_pytorch' >> ~/.bashrc
    - pip install -U 'ray[rllib]'
    - pip install simpy seaborn
'''

aws_config_str = '''# An unique identifier for the head node and workers of this cluster.
cluster_name: {}

# The maximum number of workers nodes to launch in addition to the head
# node. This takes precedence over min_workers. min_workers default to 0.
min_workers: 0

max_workers: 2

# The autoscaler will scale up the cluster faster with higher upscaling speed.
# E.g., if the task requires adding more nodes then autoscaler will gradually
# scale up the cluster in chunks of upscaling_speed*currently_running_nodes.
# This number should be > 0.
upscaling_speed: 1.0

# If a node is idle for this many minutes, it will be removed.
idle_timeout_minutes: 5


# Cloud-provider specific configuration.
provider:
    type: aws
    region: eu-west-1
    availability_zone: eu-west-1a

# How Ray will authenticate with newly launched nodes.
auth:
    ssh_user: ubuntu

# Provider-specific config for the head node, e.g. instance type.
head_node:
    InstanceType: m5.large
    ImageId: ami-017849919db4eac7c # amazon/Deep Learning AMI (Ubuntu 18.04) Version 40.0

# Provider-specific config for worker nodes, e.g. instance type.
worker_nodes:
    InstanceType: m5.large
    ImageId: ami-017849919db4eac7c # amazon/Deep Learning AMI (Ubuntu 18.04) Version 40.0
    InstanceMarketOptions:
        MarketType: spot

file_mounts: {{
    '/home/ubuntu/trainer': '~/drl-trainers/simpy/{}',
}}

# List of shell commands to run to set up nodes.
setup_commands:
    - pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    - pip install -U 'ray[rllib]'
    - pip install simpy seaborn

'''

def scaler_config(cloud_provider: str, cluster_name: str, trainer_path: str):
    cluster_map = {ord(c): None for c in '_-%&?»«!@#$'}
    if cloud_provider == "azure":
        return azure_config_str.format(cluster_name.translate(cluster_map), trainer_path)
    if cloud_provider == "aws":
        return aws_config_str.format(cluster_name.translate(cluster_map), trainer_path)
    raise "Invalid Cloud Provider '{}'. Available Cloud Providers are ['azure','aws']".format(cloud_provider)

