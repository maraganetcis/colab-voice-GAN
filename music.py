# Google Drive 체크포인트 시스템이 포함된 전체 코드

# 1. 필수 라이브러리 설치
!pip install tensorflow==2.13.0
!pip install librosa matplotlib scipy numba pydub
!apt-get install -y libsndfile1

# 2. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

import os
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import IPython.display as ipd
import zipfile
import shutil
import json
from datetime import datetime

# 3. 체크포인트 관리 클래스
class CheckpointManager:
    def __init__(self, workspace_path):
        self.workspace_path = workspace_path
        self.checkpoint_file = os.path.join(workspace_path, 'training_checkpoint.json')
    
    def save_checkpoint(self, epoch, d_loss, g_loss, training_history):
        """학습 상태 저장"""
        checkpoint = {
            'last_epoch': epoch,
            'd_loss': d_loss,
            'g_loss': g_loss,
            'training_history': training_history,
            'timestamp': datetime.now().isoformat(),
            'total_training_time': '계산 필요'  # 실제 구현에서는 시간 계산 추가
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"💾 체크포인트 저장: Epoch {epoch}")
    
    def load_checkpoint(self):
        """학습 상태 불러오기"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"📥 체크포인트 불러옴: Epoch {checkpoint['last_epoch']}")
            return checkpoint
        else:
            print("📝 새로운 체크포인트 생성")
            return None
    
    def get_last_saved_model(self):
        """가장 최근 모델 찾기"""
        model_path = os.path.join(self.workspace_path, 'saved_models')
        if not os.path.exists(model_path):
            return None
        
        model_files = [f for f in os.listdir(model_path) if f.startswith('generator_epoch_')]
        if not model_files:
            return None
        
        # 가장 높은 epoch 번호 찾기
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
        last_epoch = max(epochs)
        return last_epoch

# 4. 작업 공간 설정
def setup_workspace():
    base_path = '/content/drive/MyDrive/music_ai_project'
    folders = ['zip_files', 'extracted_music', 'generated_music', 'saved_models', 'training_logs']
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
    
    return base_path

workspace_path = setup_workspace()
checkpoint_manager = CheckpointManager(workspace_path)

# 5. 데이터 프로세서 (ZIP 처리 포함)
class MusicDataProcessor:
    def __init__(self, sr=22050, duration=4.0, n_mels=128):
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.target_length = int(sr * duration)
    
    def load_and_process_data(self):
        """데이터 로드 및 처리 (ZIP 파일에서)"""
        # ZIP 파일 처리 코드 (이전과 동일)
        # ... (이전 ZIP 처리 코드 유지)
        
        # 테스트 데이터 (실제 데이터 없을 때)
        test_spectrograms = []
        for i in range(200):
            t = np.linspace(0, 4, 22050*4)
            base_freq = 220 + (i % 8) * 50
            harmonic = 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            audio = 0.5 * np.sin(2 * np.pi * base_freq * t) + harmonic
            audio += 0.1 * np.random.normal(0, 1, len(audio))
            
            spec = self.audio_to_mel_spectrogram(audio)
            test_spectrograms.append(spec)
        
        return np.array(test_spectrograms)
    
    def audio_to_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mels, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        mel_spec_db = mel_spec_db * 2 - 1
        return mel_spec_db.astype(np.float32)[..., np.newaxis]
    
    def spectrogram_to_audio(self, spectrogram):
        if len(spectrogram.shape) == 4:
            spectrogram = spectrogram[0, :, :, 0]
        else:
            spectrogram = spectrogram[:, :, 0]
        
        spectrogram = (spectrogram + 1) / 2
        spectrogram = librosa.db_to_power(spectrogram * 80 - 80)
        
        audio = librosa.feature.inverse.mel_to_audio(
            spectrogram, sr=self.sr, hop_length=512, win_length=1024, n_iter=32
        )
        return audio

# 6. GAN 모델 (체크포인트 지원)
class MusicGANWithCheckpoint:
    def __init__(self, latent_dim=100, workspace_path=None):
        self.latent_dim = latent_dim
        self.workspace_path = workspace_path
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.d_losses = []
        self.g_losses = []
        
        # 체크포인트에서 상태 복원
        self.restore_training_state()
    
    def build_generator(self):
        # 기존 generator 코드
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(self.latent_dim,)),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Reshape((8, 8, 512)),
            tf.keras.layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2DTranspose(1, (5,5), strides=(1,1), padding='same', use_bias=False, activation='tanh')
        ])
        return model
    
    def build_discriminator(self):
        # 기존 discriminator 코드
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(128,128,1)),
            tf.keras.layers.LeakyReLU(0.2), tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2), tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(256, (5,5), strides=(2,2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2), tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(512, (5,5), strides=(2,2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2), tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(), tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def restore_training_state(self):
        """이전 학습 상태 복원"""
        checkpoint = checkpoint_manager.load_checkpoint()
        last_model_epoch = checkpoint_manager.get_last_saved_model()
        
        if last_model_epoch is not None:
            print(f"🔄 이전 모델에서 이어서 학습: Epoch {last_model_epoch}")
            self.load_models(last_model_epoch)
            
            if checkpoint:
                self.d_losses = checkpoint.get('training_history', {}).get('d_losses', [])
                self.g_losses = checkpoint.get('training_history', {}).get('g_losses', [])
                print(f"📊 이전 학습 기록: {len(self.d_losses)} epochs")
        
        return last_model_epoch if last_model_epoch else 0
    
    def train_step(self, real_spectrograms):
        # 기존 train_step 코드
        batch_size = tf.shape(real_spectrograms)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_spectrograms = self.generator(random_latent_vectors, training=True)
            real_output = self.discriminator(real_spectrograms, training=True)
            fake_output = self.discriminator(generated_spectrograms, training=True)
            
            disc_real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            disc_fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
        return disc_loss, gen_loss
    
    def generate_music(self, num_samples=1):
        random_latent_vectors = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.generator(random_latent_vectors, training=False)
    
    def save_models(self, epoch):
        """모델 저장"""
        model_path = os.path.join(self.workspace_path, 'saved_models')
        self.generator.save_weights(os.path.join(model_path, f'generator_epoch_{epoch}.h5'))
        self.discriminator.save_weights(os.path.join(model_path, f'discriminator_epoch_{epoch}.h5'))
    
    def load_models(self, epoch):
        """모델 불러오기"""
        model_path = os.path.join(self.workspace_path, 'saved_models')
        self.generator.load_weights(os.path.join(model_path, f'generator_epoch_{epoch}.h5'))
        self.discriminator.load_weights(os.path.join(model_path, f'discriminator_epoch_{epoch}.h5'))

# 7. 메인 실행 함수 (체크포인트 지원)
def main_with_checkpoints():
    print("🎵 체크포인트 지원 음악 생성 AI 시작!")
    
    # 데이터 준비
    processor = MusicDataProcessor()
    real_spectrograms = processor.load_and_process_data()
    
    # GAN 모델 생성 (자동으로 체크포인트 복원)
    music_gan = MusicGANWithCheckpoint(workspace_path=workspace_path)
    
    # 학습 파라미터
    start_epoch = len(music_gan.d_losses)  # 복원된 epoch에서 시작
    total_epochs = start_epoch + 500  # 총 500 epoch 더 학습
    batch_size = 16
    
    print(f"🚀 학습 시작: Epoch {start_epoch} → {total_epochs}")
    
    try:
        for epoch in range(start_epoch, total_epochs):
            epoch_d_loss = []
            epoch_g_loss = []
            
            # 데이터 셔플 및 배치 학습
            indices = np.random.permutation(len(real_spectrograms))
            shuffled_spectrograms = real_spectrograms[indices]
            
            for i in range(0, len(shuffled_spectrograms), batch_size):
                batch = shuffled_spectrograms[i:i+batch_size]
                if len(batch) == batch_size:
                    d_loss, g_loss = music_gan.train_step(batch)
                    epoch_d_loss.append(d_loss.numpy())
                    epoch_g_loss.append(g_loss.numpy())
            
            # 손실 기록
            avg_d_loss = np.mean(epoch_d_loss) if epoch_d_loss else 0
            avg_g_loss = np.mean(epoch_g_loss) if epoch_g_loss else 0
            
            music_gan.d_losses.append(avg_d_loss)
            music_gan.g_losses.append(avg_g_loss)
            
            # 진행 상황
            if epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | D: {avg_d_loss:.4f} | G: {avg_g_loss:.4f}")
                
                # 샘플 생성
                try:
                    generated_specs = music_gan.generate_music(1)
                    audio = processor.spectrogram_to_audio(generated_specs[0])
                    audio = audio / np.max(np.abs(audio))
                    
                    print("🔊 생성된 음악:")
                    ipd.display(ipd.Audio(audio, rate=22050))
                    
                    # 저장
                    output_path = os.path.join(workspace_path, 'generated_music', f'epoch_{epoch}.wav')
                    wavfile.write(output_path, 22050, audio)
                    
                except Exception as e:
                    print(f"⚠️ 샘플 생성 오류: {e}")
            
            # 체크포인트 저장 (매 10 epoch마다)
            if epoch % 10 == 0:
                training_history = {
                    'd_losses': music_gan.d_losses,
                    'g_losses': music_gan.g_losses
                }
                checkpoint_manager.save_checkpoint(epoch, avg_d_loss, avg_g_loss, training_history)
                music_gan.save_models(epoch)
        
        print("🎉 학습 완료!")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        print("💾 마지막 상태 저장 중...")
        # 오류 발생시 현재 상태 저장
        training_history = {
            'd_losses': music_gan.d_losses,
            'g_losses': music_gan.g_losses
        }
        checkpoint_manager.save_checkpoint(len(music_gan.d_losses)-1, 
                                         music_gan.d_losses[-1] if music_gan.d_losses else 0,
                                         music_gan.g_losses[-1] if music_gan.g_losses else 0,
                                         training_history)
        
        if len(music_gan.d_losses) > 0:
            music_gan.save_models(len(music_gan.d_losses)-1)

# 8. 실행
if __name__ == "__main__":
    print("TensorFlow 버전:", tf.__version__)
    print("GPU 사용 가능:", len(tf.config.list_physical_devices('GPU')) > 0)
    
    main_with_checkpoints()