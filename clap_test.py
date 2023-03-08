import argparse
import laion_clap
import torch
from dataset.dataset import get_wds_loader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main(args):
    clap_model = laion_clap.CLAP_Module(enable_fusion=True).to("cuda")
    clap_model.load_ckpt(model_id=3)

    names = []

    train_dl = get_wds_loader(
        batch_size=args.batch_size,
        s3_url_prefix=None,
        sample_size=args.sample_size,
        names=names,
        sample_rate=args.sample_rate,
        num_workers=args.num_workers,
        recursive=True,
        random_crop=True,
        epoch_steps=10,
    )

    all_embeddings = []

    for i, batch in enumerate(iter(train_dl)):
        print(f"Batch {i}")
        audios, jsons, timestamps = batch
        audios = audios[0].to("cuda")
        
        audios = torch.mean(audios, dim=1)

        clap_audio_embeds = clap_model.get_audio_embedding_from_data(audios.cpu().numpy())
        all_embeddings.append(clap_audio_embeds)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)

    plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])
    plt.savefig("pca_chart.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-b','--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('-s','--sample_size', help='Sample size', type=int, default=480000)
    parser.add_argument('-r','--sample_rate', help='Sample rate', type=int, default=48000)
    parser.add_argument('-w','--num_workers', help='Number of workers', type=int, default=12)
    args = parser.parse_args()

    main(args)