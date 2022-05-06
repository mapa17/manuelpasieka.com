---
title: "Deploying a Machine Learning development environment in less than 5 minutes"
date: 2022-04-21T00:00:00Z
draft: false
---

In this blog post, you are going to learn, how to deploy an AWS instance
with a python Machine Learning development environment using [Terraform](https://www.terraform.io/intro).
So you will be able to deploy your next cloud based Machine Learning environment in less than 5 minutes.

This post is most useful for people that have at least a basic understanding of
the applied technologies (AWS, Terraform, Unix), so they can adapt the provided
template for their own projects. For those that have never worked with these
technologies I tried to include references to introduction material.

All scripts used in this tutorial are available on github as part of the [onomatico](https://github.com/mapa17/onomatico) project.

## Pre-Requisites
To be able to use what you learned in this tutorial, and run the automation scripts provided, you need 
the following.

* AWS account and user credentials: You can create an AWS user for free [here](https://portal.aws.amazon.com/billing/signup) and in addition, you need to have a private/public key pair that is associated with your AWS user. On [how to create key pairs look](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and how to [associate it with your AWS user](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html#how-to-generate-your-own-key-and-import-it-to-aws). **Note: If you want to run p2.xlarge instances make sure that you have increased your quota to at least 4. See [How to increase vCPU limits](https://aws.amazon.com/de/premiumsupport/knowledge-center/ec2-on-demand-instance-vcpu-increase/)**

* **Conda Environment**: I highly recommend using a virtual environment like conda to install all tools and libraries that you are working with. Get miniconda installation for your platform [here](https://docs.conda.io/en/latest/miniconda.html)

* **Local Terraform Installation**: In your virtual conda environment install terraform from the conda-forge repo with ```conda install -c conda-forge terraform```
**Note: I recently switched to an M1 Pro and unfortunately there seems to be be no Terraform package neither in conda nor any dep package for the ARM64 platform at the time of this writing. Luckily HashiCorp has you covered and provides a dedicated ARM64 binary you can get from their [official download page](https://www.terraform.io/downloads). First try to install terraform using conda, and if that fails download it manually** 

* **Example project**: The example project that contains the terraform and other scripts you can with git directly (install with conda if you should not have git). ```git clone https://github.com/mapa17/onomatico.git```

## Overview
Modern Infrastructure management is relying heavily on the concept of [infrastructure as code](https://stackify.com/what-is-infrastructure-as-code-how-it-works-best-practices-tutorials/) and its tooling. In our case, we are going to make use of [Terraform](https://www.terraform.io/intro) as an "infrastructure as code" software framework.

This means that we use a terraform script (part of the example project in [onomatico/deployment/aws_instance.tf](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/aws_instance.tf)) that is executed locally, to defines what infrastructure (i.e. instance) we want to create on AWS. This tool basically performs in an programmatic way the infrastructure and system configuration that you would "normally" do per hand when using the AWS web console. 

To configure that instance we are going to use a bash script [onomatico/deployment/setup_instance.sh](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/setup_instance.sh) that is executed on the remote instance and is preparing a conda environment and installs the base requirements for our example project.

Now the overall picture is clear, let's dive into the details.

## Choosing the right cloud instance
In this tutorial, we are making use of AWS EC2 as the cloud provider of our choice. But don't worry if you are afraid to be vendor locked, as one of the benefits of a platform like Terraform is its multi-cloud capability, meaning that one can deploy and manage infrastructure on different cloud providers by, in the best case, only adapting some of your terraform scripts slightly.

Something that Terraform does not do for you, is to select the best **instance type** and **AMI** (i.e. system image) for your project. 
The instance type should be selected based on the requirement of computing capacity, memory, and if a GPU is needed or not. 
As this tutorial has the goal to provide you with a Machine Learning development environment, I opted for an [p2.xlarge](https://aws.amazon.com/ec2/instance-types/p2/) that comes with 62 gB Memory, 4 vCPU's and a GPU (Nvidia Tesla K80). An alternative instance type is a stronger but as well more expansive p3.2xlarge instance.

I highly recommend checking [instances.vantage.sh](https://instances.vantage.sh/) to compare different instances and their configurations as well as to get an estimate of their costs.

Once you selected the instance type of your choice you have to decide on the system image that will be used on the new instance. As part of this tutorial, I chose Amazons own [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) that comes pre-installed with an AWS optimized linux installation, and the nvidia drivers (CUDA) so ML Frameworks like tensorflow or pytorch can make use of GPU acceleration.

There is unfortunately no simple way to identify the AMI ID one needs to select the correct system image. I recommend going to the AWS web console and using AMI Catalog which is part of the EC2 web interface and searching for "Deep Learning Base AMI (Amazon Linux 2)" (the AMI ID is part of the description).

## Defining and using the Terraform configuration
We define the instance and how it should be configured using the following Terraform configuration file ([onomatico/deployment/setup_instance.sh](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/setup_instance.sh))

```python
################################ User variables ################################

# Path to private and public key used with AWS
variable "prv_key_path" {
  default = "~/.ssh/aws_dev"
}

variable "pub_key_path" {
  default = "~/.ssh/aws_dev.pub"
}

# AWS Instance type that should be deployed
variable "instance_type"{
  #default = "p2.xlarge"
  default = "t2.small"
}

# AWS AMI Image ID
variable "ami"{
  default = "ami-0b879110efb09396b"
}

# AWS Region we want the instance to run in
provider "aws" {
  region = "eu-central-1"
}


########################### Terraform Configuration ############################

terraform {
  required_version = ">= 1.0.7"
}

resource "aws_key_pair" "pub-key" {
  key_name   = "pub-key"
  public_key = file("${var.pub_key_path}")
}

# Define an AWS instance using the usr variables from above
# Make user of the user_data (ie.e cloud init) option to define a bash script
# to configure the instance at the first boot
resource "aws_instance" "instance" {
  ami                         = "${var.ami}"
  instance_type               = "${var.instance_type}"
  key_name                    = aws_key_pair.pub-key.id
  vpc_security_group_ids      = [aws_security_group.sg.id]
  associate_public_ip_address = true
  user_data = "${file("setup_instance.sh")}"

  root_block_device {
    volume_size           = 60
    delete_on_termination = true
  }

  tags = {
    Name = "AWS_ML_Instance"
  }
}

# Our user_data script will be executed as part of cloud-init final
# Block terraform until cloud-init has finished executing our script and the instance is ready.
# Create a special null_resource that will use remote-exec to wait until cloud-init has finished
resource "null_resource" "cloud_init_wait" {
  connection {
    host        = "${aws_instance.instance.public_ip}"
    user        = "ec2-user"
    private_key = "${file("${var.prv_key_path}")}"
    #script_path = "/tmp/cloud_init_wait.sh"
    timeout     = "10m"
  }
  provisioner "remote-exec" {
    inline = ["sudo cloud-init status --wait"]
  }
  depends_on = [aws_instance.instance]
}

# Define a basic security group that restricts inbound access to ssh but allows
# all outgoing access
resource "aws_security_group" "sg" {
  name        = "my_security_group"
  description = "Only allow inbound ssh access"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }


  tags = {
    Name = "AWS_ML_DEV_INSTANCE"
  }
}

# On termination of the script set an output varialbe containing the public ip
# of the instance so we can access it using ssh
output "instance-public-ip" {
  value = aws_instance.instance.public_ip
}
```

The configuration has the following main parts:
* **User variables** Those settings are the one you would want to change based on your system and use to apply the script in another project
* **AWS Instance** configuration that applies the user variables to define the instance
* a **dummy terraform resource provider** that will take care to block the terraform deployment while **cloud init** is used to prepare the instance
* a simple **aws security group** that restricts access to the machine for security reasons

To use the script you needs to apply the following commands
* `terraform init`: Running init in the project directory where the configuration is located will create several (hidden) folders and files that terraform uses to build a deployment plan for your configuration and keep track of the executed configurations to be able to tear them down after you don't need them anymore.
* `terraform plan`: will launch the deployment planning process that will read your configurations (all files with the extension .tf), analyze them and create a deployment plan with the desired state that terraform will try to achieve. You can store this plan and execute it later for reproducibility.
* `terraform apply`: is creating a plan if none exists and will execute it, accomplishing the actual creation of instances and resources.
* `terraform destroy`: will remove any running instances and resources defined in your configuration. 

But before you can use terraform, you have to provide it with essential AWS credentials through environment variables.

In particular your [AWS Access and Secret key](https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html) that you can expose to Terraform through the following environment variables by executing:

```bash
export AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXX
export AWS_DEFAULT_REGION=XXXXXXXXXXX
```

The last thing that is missing is the location of your public and private key pair that is associated with your AWS user (ie. the user that is specified in the AWS access and secret key with the permissions to create instances and connect to them). To do so, you can use the two user variables **prv_key_path** and **pub_key_path** at the top of the Terraform configuration file.

## Configuring the cloud instance for an ML Project
After Terraform has created the AWS instance it runs the select AMI image. In our case is an AWS optimized Linux installation with the Nvidia drivers and libraries (i.e. Cuda) to support GPU hardware acceleration, but nothing more.

To be able to use this instance as an ML development machine we need our ML libraries and tooling installed. As always I recommend conda + pip as the virtual environment and package management systems of your choice.

To automatize this part of the system configuration I have created a bash script (i.e. [onomatico/deployment/setup_instance.sh](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/setup_instance.sh)) that will download miniconda, install a conda environment with the name `mldev`, will git clone our experiment project, and will install all its dependencies.

Once this is done, you can ssh into the instance, activate the conda environment (with `conda activate mldev`), and enjoy a Pytorch development environment with GPU support and many of the most common ML libraries.

Let's have a quick look at that setup script [onomatico/deployment/setup_instance.sh](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/setup_instance.sh)

```bash
#!/bin/sh
# Note: aws cloud-init requires the shebang to be `#!/bin/sh` nothing else will work!


################################### User Settings ##############################

# Define user variables 
USER="ec2-user"
HOME="/home/ec2-user"
CONDA="${HOME}/miniconda3/bin/conda"
CONDA_ACTIVATE="${HOME}/miniconda3/bin/activate"
PROJECT="https://github.com/mapa17/onomatico.git"
PROJECT_HOME="${HOME}/onomatico"

################################### Logging ####################################

# Create a log message that will be visible in /var/log/cloud-init-output.log to verify the exeuction of this script
echo "## Executing instance configuration script!"

# Terminate script on any error
set -e

# Log all output to file
exec > /tmp/setup_instance.log                                                                      
exec 2>&1

######################## Setup conda and project dependencies ##################

# Download miniconda
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh

# Install as ec2-user miniconda in batch mode
sudo -u ${USER} -- bash -c "cd ${HOME} && echo -n 'Running as ' && who && bash /tmp/miniconda.sh -b && ${CONDA} init bash"

# Create env with support for poetry (will download a lot of other stuff)
sudo -u ${USER} -- bash -c "cd ${HOME} && ${CONDA} create -n mldev poetry -y"

# Clone the repo and install dependencies
sudo -u ${USER} -- bash -c "cd ${HOME} && source ${CONDA_ACTIVATE} mldev && git clone ${PROJECT} ${PROJECT_HOME} && cd ${PROJECT_HOME} && poetry install" 

```

Similar to the Terraform configuration it starts with a bunch of project-specific user settings, like the path to the github repository containing our project and some AWS specific settings like the username and home directory of the default user.

Next are some settings that will make sure to log all output created from this script to be written to `/tmp/setup_instance.log` for debugging in case of errors. To detect problems easier, we let this script stop and fail as soon as any single command fails, by setting `set -e`.

The last part of the script is executing a series of shell commands that download, install and configure conda, plus the project and its dependencies. We make sure to run those commands as a normal user, as the script itself is executed by cloud-init as root.

## Making sure that terraform waits for the instance to complete its setup
The bash script that we use to configure the newly created instance is executed through AWS [cloud-init](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html) 
by specifying the content of the script as the parameter `user_data` in the AWS instance configuration in Terraform.

This will cause the creation of a bash script on the new instance and it's executing as part of `clout-init final` (A system process that is run during system boot). Unfortunately, the execution of this step can take quite some time. The first time the instance boots, cloud-init is executing besides our script, several other AWS pre-configured system modifications and performs an system update.

This system boot process is out of the scope of Terraform, and so the Terraform deployment process will not wait for cloud-init nor has it means to monitor its progress. Instead, Terraform will simply create the instance and provide the script to cloud-init, assuming its job is done.

This behavior can be acceptable in a case where you deploy some new cloud resources, like compute notes in a cluster, storage 
pods for later use, or any asynchronous use case where you have other measures in place that make use once the new resource
is not only deployed but actually useable. In our use case though, you want to be able to know when your new ML instance
is ready to be used, and therefore we want to reflect the system boot status during the Terraform deployment. One easy way
to solve this, is to make Terraform block the deployment until the system has booted and is configured completely.

To make sure that `terraform apply` will only terminate once cloud-init has finished, we introduce a [remove-exec terraform provider](https://www.terraform.io/language/resources/provisioners/remote-exec) named `cloud_init_wait` that is using ssh on the remote instance and is executing `sudo cloud-init status --wait` which blocks until cloud-init finishes its execution.

Once this modified Terraform configuration terminates executing, we know that the machine is ready and we can access it through ssh ourselves. The `public ip` of our newly created instance is contained at the end of the output generated with `terraform apply`.

Example:
```bash
% terraform apply 
...
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .
null_resource.cloud_init_wait (remote-exec): .

null_resource.cloud_init_wait (remote-exec): status: done
null_resource.cloud_init_wait: Creation complete after 4m46s [id=527681618375576572]

Apply complete! Resources: 4 added, 0 changed, 0 destroyed.

Outputs:

instance-public-ip = "3.67.40.126"
```

You can then, access the machine with the default AWS user and your private ssh key

```bash
ssh -i ~/.ssh/aws_dev ec2-user@3.67.40.126
```

## Putting it all together

1. Clone the repository
2. Set the AWS environment variables (Credentials)
3. Adapt the user settings in `aws_instance.tf` to your project (i.e. set path to pub/private keys, configure instance type, AMI)
4. Adapt the user settings in `setup_instance.sh` (i.e. provide the URL to your repo and add any required setup instructions)
5. Initialize Terraform in the within `onomatico/deployment` with `terraform init`
```bash
git clone git@github.com:mapa17/onomatico.git
```

## How to resolve errors on the way
The infrastructure automation you learned in this tutorial has two main sources of possible errors.

The Terraform configuration and the bash script that is used to configure the project environment.

Terraform will do its best to inform you about errors in your configuration when you execute `terraform plan` and if you don't wander too far away from the provided template you should be fine. Besides googling for a particular error message (always the best and hopefully your first approach) the official [Terraform documentation](https://www.terraform.io/intro) is your best source of information about the Terraform API and details about different configuration parameters.

Far more difficult to identify are errors in the system configuration and the execution of the bash script. Any errors that happen during the executing with cloud-init will cause Terraform to quit with a rather cryptic error message similar to the following

```bash
null_resource.cloud_init_wait (remote-exec): status: error
╷
│ Error: remote-exec provisioner error
│
│   with null_resource.cloud_init_wait,
│   on aws_instance.tf line 64, in resource "null_resource" "cloud_init_wait":
│   64:   provisioner "remote-exec" {
│
│ error executing "/tmp/terraform_823990194.sh": Process exited with status 1
```

If this is the case, you have to manually investigate the cloud init log files on the new instance in `/var/log/cloud-init-output.log` or the script log that is written to `/tmp/setup_instance.log`.

## Where to go from here
Congratulations!  You have now, an AWS instance running and configured and can go ahead to develop some amazing new AI-powered applications!

Obviously, you want to take the scripts shown here as a template for your own projects. Doing so you should reflect and if necessary adjust the following configuration settings
* instance type ([onomatico/deployment/aws_instance.tf](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/aws_instance.tf))
* AMI image ([onomatico/deployment/aws_instance.tf](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/aws_instance.tf))
* git project repository ([onomatico/deployment/setup_instance.sh](https://github.com/mapa17/onomatico/blob/17519ca4f11667a4251f21746e10f99fd2cec253/deployment/setup_instance.sh))

Good luck!

## Where not to go from here
The goal of this tutorial is to provide you with a template to use for quick deployment of a new ML development environment in the cloud.

It's not intended to be used for the deployment of production infrastructure or to build an ML workflow that would be launching instances, training models, and doing testing or prediction automatically.

You can build all those nice things by yourself and make use of Terraform to manage your cloud infrastructure, but first I would recommend you to have a look at tools like [KubeFlow](https://www.kubeflow.org/docs/started/introduction/) for your ML Pipelines.

## Conclusion
With this tutorial, you have hopefully reached a basic understanding of how to use Tools like Conda, Terraform, and AWS EC2 as well as a Template to start your next ML development project on an AWS instance.

If you have some questions or comments write me at contact@manuelpasieka.com