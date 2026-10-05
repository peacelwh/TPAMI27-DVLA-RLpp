"""Episode samplers shared by meta-tuning, evaluation and pre-training."""
import numpy as np
import torch
import torch.utils.data


class EpisodeSampler:
    """Yield index batches of ``n_cls * n_per`` images forming an N-way episode.

    With ``fix_seed=True`` the episode list is generated once from seed 0 so that
    evaluation and ablations use identical episodes.
    """

    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            self.m_ind.append(torch.from_numpy(ind))

        if self.fix_seed:
            state = np.random.get_state()
            np.random.seed(0)
            self.cached_batches = torch.stack([self._draw() for _ in range(self.n_batch)])
            np.random.set_state(state)

    def _draw(self):
        batch = []
        classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
        for c in classes:
            l = self.m_ind[c]
            pos = np.random.choice(range(len(l)), self.n_per, False)
            batch.append(l[pos])
        return torch.stack(batch).reshape(-1)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            yield self.cached_batches[i_batch] if self.fix_seed else self._draw()


class RepeatSampler:
    """Repeat every mini-batch ``repeat`` times (repeated augmentation for pre-training)."""

    def __init__(self, dataset, batch_size, repeat):
        self.batch_size = batch_size // repeat
        self.repeat = repeat
        self.sampler = torch.utils.data.RandomSampler(dataset)
        self.drop_last = True

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch * self.repeat
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
