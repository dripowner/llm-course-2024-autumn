from tqdm.auto import tqdm
from .compute_reward import compute_reward

def eval_reward_model(reward_model, reward_tokenizer, test_dataset, target_label, device='cpu'):
	"""
	Evaluate the performance of a reward model by comparing reward scores for chosen and rejected reviews. 

	This function selects reviews from a test dataset based on a target label and evaluates the reward model's
	ability to assign higher scores to chosen reviews compared to rejected ones. The evaluation is performed
	in batches for efficiency.
	Note that reward scores are compared on corresponding chosen and rejected reviews: 
		chosen_reviews[0] vs rejected_reviews[0], 
		chosen_reviews[1] vs rejected_reviews[1],
		etc.

	Parameters:
	reward_model: The model used to compute the reward scores
	reward_tokenizer: The tokenizer for reward_model
	tes_dataset: test Dataset
	target_label (0 or 1): The label used to select chosen reviews. Reviews with this label are considered chosen,
				  while others are considered rejected.
	device (str, optional): The device on which the computation should be performed. Default is 'cpu'.

	Returns:
	float: The accuracy of the reward model, calculated as the proportion of times the model assigns a higher
		   reward score to the chosen review compared to the rejected review.

	Example:
	>>> accuracy = eval_reward_model(my_reward_model, my_reward_tokenizer, test_data, target_label=1)
	>>> print(f"Model accuracy: {accuracy:.2%}")
	"""

	chosen_reviews = [x['text'] for x in test_dataset if x['label'] == target_label]
	rejected_reviews = [x['text'] for x in test_dataset if x['label'] != target_label]

	if len(chosen_reviews) != len(rejected_reviews):
		min_len = min(len(chosen_reviews), len(rejected_reviews))
		chosen_reviews = chosen_reviews[:min_len]
		rejected_reviews = rejected_reviews[:min_len]

	assert len(chosen_reviews) == len(rejected_reviews)

	correct_answers_count = 0
	for chosen_review, rejected_review in tqdm(zip(chosen_reviews, rejected_reviews)):

		# rewards = compute_reward(reward_model, reward_tokenizer, [chosen_review, rejected_review], device=device)
		rewards = compute_reward(reward_model, reward_tokenizer, [chosen_review, rejected_review])

		chosen_reward = rewards[0].item()
		rejected_reward = rewards[1].item()
		
		if chosen_reward > rejected_reward:
			correct_answers_count += 1

	return correct_answers_count / len(chosen_reviews)