from imblearn import under_sampling
import numpy as np
import argparse

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str, required=True, help='Embeddings')
    parser.add_argument('--labels', type=str, required=True, help='Labels')
    parser.add_argument('--output', type=str, required=True, help='Output file')
    args = parser.parse_args()

    x = np.loadtxt(args.embeddings)
    y = np.loadtxt(args.labels)

    print('Before:', np.bincount(y.astype(int)))
    sampler = under_sampling.NearMiss(version=3)
    x_res, y_res = sampler.fit_resample(x, y)
    print('After:', np.bincount(y_res.astype(int)))

    np.savetxt(args.output, sampler.sample_indices_, fmt='%d')
    print('Done')



