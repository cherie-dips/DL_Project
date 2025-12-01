import torch, time
from Mobile_Hi_SAM.models.mobile_encoder import MobileSAMEncoder

def main():
    encoder = MobileSAMEncoder()
    x = torch.randn(1, 3, 1024, 1024)

    t0 = time.time()
    y = encoder(x)
    t1 = time.time()

    print("Output shape:", y.shape)
    print("Time taken:", t1 - t0, "seconds")

if __name__ == "__main__":
    main()

