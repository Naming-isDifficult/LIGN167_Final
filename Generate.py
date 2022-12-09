from WaveNet.Generator import Generator
from DataLoader.AudioDataLoader import mu_law_decode, mu_law_encode
import librosa

if __name__ == '__main__':

    #extract seed from existing file
    seed, _ = librosa.load('seed.wav', sr=16000, duration=1.1)

    print(seed.shape)

    generator = Generator('Model\WaveNet_Model_2022-12-04\wavenet_step37820_loss5.3806.model',\
                          'Model\WaveNet_Model_2022-12-04\wavenet_step37820_loss5.3806.weight')

    generator.generate_samples(5, 'output.wav', seed=seed)