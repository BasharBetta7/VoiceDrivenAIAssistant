from my_langsmith.config import ls_client, log


DATASET_NAME= 'agent-evals'
DATASET_DESC= 'simple QA for our agent'

examples = [
    {
        "inputs": {"question": "What is your name?"},
        "outputs": {"answer": "My name is Aurelious; I am your personal assistant."},
    },
    {
        "inputs": {"question": "What is 12 multiplied by 8?"},
        "outputs": {"answer": "96"},
    },
    {
        "inputs": {"question": "Convert 2 hours into minutes."},
        "outputs": {"answer": "120 minutes"},
    },
]


datasets = ls_client.list_datasets()
log.info(f"Langsmith datasets fround: {[d.name for d in datasets]}")

def sync_dataset():
    if ls_client.has_dataset(dataset_name=DATASET_NAME):
        dataset = ls_client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = ls_client.create_dataset(dataset_name=DATASET_NAME, description=DATASET_DESC)

    existing = {
        example.inputs['question']: example for example in ls_client.list_examples(dataset_id=dataset.id)
    }
    for example in examples:
        question = example["inputs"]["question"]
        previous = existing.get(question)
        if previous is None:
            created = ls_client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
            )
            existing[question] = created
            log.info("Added: %s", question)
        elif previous.outputs != example['outputs']:
            ls_client.update_example(
                example_id= previous.id,
                outputs=example['outputs'],
            )
            log.info("Updated reference answer: %s", question)


if __name__ == '__main__':
    sync_dataset()


    
    