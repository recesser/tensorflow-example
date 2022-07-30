# Recesser Tensorflow Example

This example is based on a [Tensorflow tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) for classifying text with a BERT deep learning model.

First, you need to download the dataset. You can easily do that with `wget`:

```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
```

Do not unpack this. The `main.py` script will do that automatically.

You can now either train the model locally, or inside the Recesser System.

## Local Execution

You need to have Python 3 installed. Create a virtual environment and install the dependencies from
`requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Now run the script providing the location of the downloaded dataset on the command line:

```
python3 main.py aclImdb_v1.tar.gz
```

This will take a few minutes. After the execution has finished you will see the loss and the
accuracy of the model printed to the command line.

## Execution inside Recesser

After downloading the dataset from the internet you need to upload it to the Recesser system.

First, set the the address of your system as the environment variable `RECESSER_ADDR` and your token as `RECESSER_TOKEN`:

```bash
export RECESSER_ADDR="<address of your system>"
export RECESSER_TOKEN="<your token>"
```

Then upload the data:

```bash
rcssr artifact upload aclImdb_v1.tar.gz
```

If the file has not changed, the computed handle in the workflow description `recesser.yaml` will be
correct.

Then, register this repository with the system:

```bash
rcssr repository add "recesser/tensorflow-example"
```

The CLI will generate an SSH key, register it inside the Recesser system and print it out on the
command line. You need to copy this key and set it as a read-only [deploy
key](https://docs.github.com/en/developers/overview/managing-deploy-keys) on your GitHub repository.

Now after a short while (less than 1min on the default setting), Recesser should have picked up your
repository and started training the machine learning model.
