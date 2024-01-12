# This is a sample Python script.
import torch
import matplotlib.pyplot as plt

from dataset import LFWDataset

if __name__ == '__main__':
    # let's create a DataLoader to easily iterate over this dataset
    test_data = LFWDataset(download=False, base_folder='lfw_dataset', transforms=None)
    bs = 4
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        imgs = sample_batched[0]
        segs = sample_batched[1]

        rows, cols = bs, 2
        figure = plt.figure(figsize=(10, 10))

        for i in range(0, bs):
            figure.add_subplot(rows, cols, 2 * i + 1)
            plt.title('image')
            plt.axis("off")
            plt.imshow(imgs[i].numpy().transpose(1, 2, 0))

            figure.add_subplot(rows, cols, 2 * i + 2)
            plt.title('seg')
            plt.axis("off")
            plt.imshow(segs[i].numpy().transpose(1, 2, 0), cmap="gray")
        plt.show()
        # display the first 3 batches
        if i_batch == 2:
            break

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
