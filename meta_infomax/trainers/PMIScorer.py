from collections import OrderedDict, Counter

import itertools
import numpy as np
from tqdm.notebook import tqdm

from meta_infomax.datasets import utils
from meta_infomax.datasets.fudan_reviews import MultiTaskDataset, DATASETS


class PMIScorer:

    def __init__(self, tokenizer, keep_datasets):
        data = MultiTaskDataset(tokenizer, split="all", keep_datasets=DATASETS, validation_size=0)

        datasets = {f"{ds}_{lb}": data.subset_datasets(domains=[ds], label=lb)[ds]
                    for ds in DATASETS for lb in [0, 1]}
        datasets["all_data"] = data

        data_test = MultiTaskDataset(tokenizer, split="train", keep_datasets=keep_datasets, validation_size=0)
        self.datasets_test = {f"{ds}_{lb}": data_test.subset_datasets(domains=[ds], label=lb)[ds]
                              for ds in keep_datasets for lb in [0, 1]}

        get_counter = lambda dataset: Counter(
            list(itertools.chain.from_iterable(
                # it is a list of lists that needs to be flattened
                [dataset[data_idx]['text'].split() for data_idx in range(len(dataset))])))

        self.counters = {ds_name: get_counter(ds) for (ds_name, ds) in datasets.items()}
        self.lengths = {ds_name: len(ds) for (ds_name, ds) in datasets.items()}

    def get_PMI(self, word, domain):
        joint_prob = self.counters[domain][word]
        word_prob = self.counters["all_data"][word]
        domain_prob = self.lengths[domain] / self.lengths["all_data"]
        return np.log(joint_prob / (domain_prob * word_prob))

    def get_domain_PMIs(self, domain):
        pmi_dict = {word: self.get_PMI(word, domain) for word in self.counters[domain].keys()}
        return OrderedDict(sorted(pmi_dict.items(), key=lambda kv: -kv[1]))

    def score_samples(self, domain):
        unique_domain_words = {}
        counter = 0
        for word, pmi_score in self.get_domain_PMIs(domain).items():
            if counter < 3000:
                if self.counters["all_data"][word] > 5 and pmi_score > 0:
                    counter += 1
                    unique_domain_words[word] = pmi_score
            else:
                break

        scores = []
        words = []
        domain_ds = self.datasets_test[domain]
        for idx in tqdm(range(len(domain_ds)), leave=False):
            tokenized = domain_ds[idx]["text"].split()
            sent_scores = []
            interesting_words = []
            for token in tokenized:
                if token in unique_domain_words:
                    sent_scores.append(unique_domain_words[token])
                    interesting_words.append(token)
            scores.append(np.mean(sent_scores) if sent_scores else -np.inf)
            words.append(interesting_words)
        sort_idxs = np.array(scores).argsort()[::-1]

        return sort_idxs

    def sort_datasets(self):
        rankings = {ds_name: self.score_samples(ds_name) for ds_name, ds in self.datasets_test.items()}
        for ds_name, ranking in rankings.items():
            self.datasets_test[ds_name].data = self.datasets_test[ds_name].data.iloc[ranking]

        return self.datasets_test

if __name__ == "__main__":
    bert, tokenizer, _ = utils.get_transformer("bert-base-uncased")
    scorer = PMIScorer(tokenizer, ["dvd", "video"])
    sorted_ds = scorer.sort_datasets()
