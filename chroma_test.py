import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ================== 1. Chroma特征提取 ==================

def extract_chroma_features(audio_file, sr=22050):
    """提取chroma特征，只保留pitch class信息"""
    y, sr = librosa.load(audio_file, sr=sr)
    
    # 提取chroma特征 (12维，对应12个半音)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, 
        hop_length=512,
        n_fft=2048,
        tuning=0  # 可以设置调音偏移
    )
    
    # 标准化处理
    chroma = librosa.util.normalize(chroma, axis=0)
    
    return chroma  # shape: (12, time_frames)

def extract_chroma_from_array(audio_array, sr=22050):
    """从音频数组直接提取chroma特征"""
    chroma = librosa.feature.chroma_stft(
        y=audio_array, sr=sr, 
        hop_length=512,
        n_fft=2048,
        tuning=0
    )
    chroma = librosa.util.normalize(chroma, axis=0)
    return chroma

# ================== 2. 相对音高/Interval Embedding ==================

def create_interval_embedding(chroma_features, window_size=5):
    """
    创建基于音程的相对音高嵌入
    在chroma空间中计算音程，只考虑pitch class (C-B)
    """
    n_pitches, n_frames = chroma_features.shape
    interval_features = []
    
    for i in range(n_frames - window_size + 1):
        window = chroma_features[:, i:i+window_size]
        
        # 方法1: 相邻帧间的音程差异
        intervals = []
        for j in range(window_size - 1):
            # 计算相邻帧的主要音高 (0-11对应C-B)
            current_pitch = np.argmax(window[:, j])
            next_pitch = np.argmax(window[:, j+1])
            
            # 关键：只在12个半音内计算距离，忽略八度
            interval = (next_pitch - current_pitch) % 12
            intervals.append(interval)
        
        interval_features.append(intervals)
    
    return np.array(interval_features)

def create_chord_progression_embedding(chroma_features, window_size=8):
    """
    基于和弦进行的相对音高表示
    """
    n_pitches, n_frames = chroma_features.shape
    chord_progressions = []
    
    for i in range(n_frames - window_size + 1):
        window = chroma_features[:, i:i+window_size]
        
        # 提取每帧的和弦特征 (取前3个最强的音)
        chord_sequence = []
        for frame in window.T:
            top_pitches = np.argsort(frame)[-3:]  # 取最强的3个音
            # 转换为相对于根音的音程
            root = top_pitches[0]
            relative_pitches = [(p - root) % 12 for p in sorted(top_pitches)]
            chord_sequence.append(relative_pitches)
        
        chord_progressions.append(chord_sequence)
    
    return chord_progressions

def visualize_interval_extraction(chroma_features):
    """可视化从chroma到音程的转换过程"""
    print("原始chroma特征:")
    print(f"形状: {chroma_features.shape}")
    print("每一列代表一个时间帧的12个音高强度\n")
    
    # 提取每个时间帧的主导音高
    dominant_pitches = []
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for frame_idx in range(min(10, chroma_features.shape[1])):  # 只显示前10帧
        frame = chroma_features[:, frame_idx]
        dominant_pitch = np.argmax(frame)
        dominant_pitches.append(dominant_pitch)
        print(f"帧{frame_idx}: {pitch_names[dominant_pitch]} (索引{dominant_pitch}, 强度{frame[dominant_pitch]:.3f})")
    
    # 计算相邻帧之间的音程
    intervals = []
    for i in range(len(dominant_pitches) - 1):
        current_pitch = dominant_pitches[i]
        next_pitch = dominant_pitches[i + 1]
        interval = (next_pitch - current_pitch) % 12
        intervals.append(interval)
        
        print(f"{pitch_names[current_pitch]} → {pitch_names[next_pitch]}: "
              f"音程距离 = {interval} 半音")
    
    print(f"\n音程序列: {intervals}")
    return intervals

# ================== 3. 模型架构 ==================

class MusicVersionMatchingModel(nn.Module):
    def __init__(self, chroma_dim=12, interval_dim=64, hidden_dim=128, output_dim=64):
        super().__init__()
        
        # Chroma特征编码器
        self.chroma_encoder = nn.Sequential(
            nn.Linear(chroma_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        # Interval特征编码器
        self.interval_encoder = nn.Sequential(
            nn.Linear(interval_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, chroma_features, interval_features):
        # 如果输入是3D的，取平均池化
        if len(chroma_features.shape) == 3:
            chroma_features = torch.mean(chroma_features, dim=1)
        if len(interval_features.shape) == 3:
            interval_features = torch.mean(interval_features, dim=1)
            
        # 编码chroma特征
        chroma_emb = self.chroma_encoder(chroma_features)
        
        # 编码interval特征
        interval_emb = self.interval_encoder(interval_features)
        
        # 融合
        combined = torch.cat([chroma_emb, interval_emb], dim=-1)
        embedding = self.fusion(combined)
        
        return embedding

# ================== 4. 数据集类 ==================

class MusicVersionDataset(Dataset):
    def __init__(self, audio_pairs, window_size=5):
        """
        audio_pairs: list of lists, 每个内部list包含同一首歌的不同版本路径
        """
        self.data = []
        self.window_size = window_size
        
        # 准备正样本和负样本
        all_songs = []
        song_labels = []
        
        for song_id, versions in enumerate(audio_pairs):
            for version_path in versions:
                all_songs.append(version_path)
                song_labels.append(song_id)
        
        # 生成样本对
        for i in range(len(all_songs)):
            for j in range(i+1, len(all_songs)):
                label = 1 if song_labels[i] == song_labels[j] else 0
                self.data.append((all_songs[i], all_songs[j], label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        audio1_path, audio2_path, label = self.data[idx]
        
        # 提取特征
        try:
            chroma1 = extract_chroma_features(audio1_path)
            chroma2 = extract_chroma_features(audio2_path)
            
            interval1 = create_interval_embedding(chroma1, self.window_size)
            interval2 = create_interval_embedding(chroma2, self.window_size)
            
            # 统计特征 (取均值作为全局特征)
            chroma1_mean = np.mean(chroma1, axis=1)
            chroma2_mean = np.mean(chroma2, axis=1)
            interval1_mean = np.mean(interval1, axis=0) if len(interval1) > 0 else np.zeros(self.window_size-1)
            interval2_mean = np.mean(interval2, axis=0) if len(interval2) > 0 else np.zeros(self.window_size-1)
            
            # 填充到固定长度
            interval_dim = 64
            if len(interval1_mean) < interval_dim:
                interval1_mean = np.pad(interval1_mean, (0, interval_dim - len(interval1_mean)))
            else:
                interval1_mean = interval1_mean[:interval_dim]
                
            if len(interval2_mean) < interval_dim:
                interval2_mean = np.pad(interval2_mean, (0, interval_dim - len(interval2_mean)))
            else:
                interval2_mean = interval2_mean[:interval_dim]
            
            return {
                'chroma1': torch.tensor(chroma1_mean, dtype=torch.float32),
                'interval1': torch.tensor(interval1_mean, dtype=torch.float32),
                'chroma2': torch.tensor(chroma2_mean, dtype=torch.float32),  
                'interval2': torch.tensor(interval2_mean, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error processing {audio1_path} or {audio2_path}: {e}")
            # 返回零向量作为fallback
            return {
                'chroma1': torch.zeros(12, dtype=torch.float32),
                'interval1': torch.zeros(64, dtype=torch.float32),
                'chroma2': torch.zeros(12, dtype=torch.float32),
                'interval2': torch.zeros(64, dtype=torch.float32),
                'label': torch.tensor(0.0, dtype=torch.float32)
            }

# ================== 5. 训练函数 ==================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    """训练版本匹配模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for batch in train_loader:
            chroma1 = batch['chroma1'].to(device)
            interval1 = batch['interval1'].to(device)
            chroma2 = batch['chroma2'].to(device)
            interval2 = batch['interval2'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            emb1 = model(chroma1, interval1)
            emb2 = model(chroma2, interval2)
            
            # 计算相似度
            similarity = torch.cosine_similarity(emb1, emb2, dim=1)
            
            # 计算损失
            loss = criterion(similarity, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证阶段
        if val_loader:
            val_acc = evaluate_model(model, val_loader, device)
            val_accuracies.append(val_acc)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return train_losses, val_accuracies

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            chroma1 = batch['chroma1'].to(device)
            interval1 = batch['interval1'].to(device)
            chroma2 = batch['chroma2'].to(device)
            interval2 = batch['interval2'].to(device)
            labels = batch['label'].to(device)
            
            emb1 = model(chroma1, interval1)
            emb2 = model(chroma2, interval2)
            
            similarity = torch.cosine_similarity(emb1, emb2, dim=1)
            preds = (similarity > 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# ================== 6. 相似度计算和匹配 ==================

def compute_similarity(embedding1, embedding2):
    """计算两个音乐片段的相似度"""
    # 余弦相似度
    cos_sim = torch.cosine_similarity(embedding1, embedding2, dim=-1)
    return cos_sim

def find_matching_versions(query_audio, database_audios, model, threshold=0.7, window_size=5):
    """在数据库中找到匹配的版本"""
    device = next(model.parameters()).device
    model.eval()
    
    # 提取查询音频特征
    query_chroma = extract_chroma_features(query_audio)
    query_interval = create_interval_embedding(query_chroma, window_size)
    
    # 预处理特征
    query_chroma_mean = np.mean(query_chroma, axis=1)
    query_interval_mean = np.mean(query_interval, axis=0) if len(query_interval) > 0 else np.zeros(window_size-1)
    
    # 填充到固定长度
    interval_dim = 64
    if len(query_interval_mean) < interval_dim:
        query_interval_mean = np.pad(query_interval_mean, (0, interval_dim - len(query_interval_mean)))
    else:
        query_interval_mean = query_interval_mean[:interval_dim]
    
    # 转换为tensor
    query_chroma_tensor = torch.tensor(query_chroma_mean, dtype=torch.float32).unsqueeze(0).to(device)
    query_interval_tensor = torch.tensor(query_interval_mean, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 获取查询音频的embedding
    with torch.no_grad():
        query_embedding = model(query_chroma_tensor, query_interval_tensor)
    
    matches = []
    for db_audio in database_audios:
        try:
            # 提取数据库音频特征
            db_chroma = extract_chroma_features(db_audio)
            db_interval = create_interval_embedding(db_chroma, window_size)
            
            # 预处理
            db_chroma_mean = np.mean(db_chroma, axis=1)
            db_interval_mean = np.mean(db_interval, axis=0) if len(db_interval) > 0 else np.zeros(window_size-1)
            
            if len(db_interval_mean) < interval_dim:
                db_interval_mean = np.pad(db_interval_mean, (0, interval_dim - len(db_interval_mean)))
            else:
                db_interval_mean = db_interval_mean[:interval_dim]
            
            # 转换为tensor
            db_chroma_tensor = torch.tensor(db_chroma_mean, dtype=torch.float32).unsqueeze(0).to(device)
            db_interval_tensor = torch.tensor(db_interval_mean, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 获取embedding
            with torch.no_grad():
                db_embedding = model(db_chroma_tensor, db_interval_tensor)
            
            # 计算相似度
            similarity = compute_similarity(query_embedding, db_embedding)
            
            if similarity.item() > threshold:
                matches.append((db_audio, similarity.item()))
                
        except Exception as e:
            print(f"Error processing {db_audio}: {e}")
            continue
    
    return sorted(matches, key=lambda x: x[1], reverse=True)

# ================== 7. 示例和测试函数 ==================

def demonstrate_octave_invariance():
    """演示八度不变性"""
    print("=== 八度不变性演示 ===")
    
    # 示例：不同八度的相同旋律
    # 旋律1: C4-D4-E4-F4 (中央C八度)
    melody1_pitches = [0, 2, 4, 5]  # 在chroma空间中
    
    # 旋律2: C5-D5-E5-F5 (高一个八度)  
    melody2_pitches = [0, 2, 4, 5]  # 在chroma空间中仍然相同！
    
    # 计算音程序列
    intervals1 = [(melody1_pitches[i+1] - melody1_pitches[i]) % 12 
                  for i in range(len(melody1_pitches)-1)]
    intervals2 = [(melody2_pitches[i+1] - melody2_pitches[i]) % 12 
                  for i in range(len(melody2_pitches)-1)]
    
    print(f"旋律1的音程: {intervals1}")  # [2, 2, 1] (大二度, 大二度, 小二度)
    print(f"旋律2的音程: {intervals2}")  # [2, 2, 1] (完全相同!)
    print(f"是否相同: {intervals1 == intervals2}")  # True!
    
    return intervals1 == intervals2

def create_test_data():
    """创建测试数据"""
    # 创建一些合成的chroma特征用于测试
    sr = 22050
    duration = 5  # 5秒
    
    # C大调音阶
    c_major_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4-C5
    
    # 生成C大调音阶音频
    t = np.linspace(0, duration, sr * duration, False)
    audio_c_major = np.zeros_like(t)
    
    for i, freq in enumerate(c_major_freqs):
        start_idx = int(i * len(t) / len(c_major_freqs))
        end_idx = int((i + 1) * len(t) / len(c_major_freqs))
        audio_c_major[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
    
    # 生成D大调音阶音频 (移调+2半音)
    d_major_freqs = [f * (2 ** (2/12)) for f in c_major_freqs]
    audio_d_major = np.zeros_like(t)
    
    for i, freq in enumerate(d_major_freqs):
        start_idx = int(i * len(t) / len(d_major_freqs))
        end_idx = int((i + 1) * len(t) / len(d_major_freqs))
        audio_d_major[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])
    
    return audio_c_major, audio_d_major, sr

def test_chroma_extraction():
    """测试chroma特征提取"""
    print("=== 测试Chroma特征提取 ===")
    
    # 创建测试音频
    audio_c, audio_d, sr = create_test_data()
    
    # 提取chroma特征
    chroma_c = extract_chroma_from_array(audio_c, sr)
    chroma_d = extract_chroma_from_array(audio_d, sr)
    
    print(f"C大调chroma形状: {chroma_c.shape}")
    print(f"D大调chroma形状: {chroma_d.shape}")
    
    # 提取音程特征
    intervals_c = create_interval_embedding(chroma_c)
    intervals_d = create_interval_embedding(chroma_d)
    
    print(f"C大调音程特征形状: {intervals_c.shape}")
    print(f"D大调音程特征形状: {intervals_d.shape}")
    
    # 比较相似性
    if len(intervals_c) > 0 and len(intervals_d) > 0:
        similarity = np.corrcoef(intervals_c.flatten(), intervals_d.flatten())[0,1]
        print(f"音程序列相似度: {similarity:.3f}")
    
    return chroma_c, chroma_d, intervals_c, intervals_d

# ================== 8. 主函数示例 ==================

def main():
    """主函数示例"""
    print("=== Music Version Matching with Chroma Features ===")
    
    # 测试八度不变性
    demonstrate_octave_invariance()
    print()
    
    # 测试特征提取
    chroma_c, chroma_d, intervals_c, intervals_d = test_chroma_extraction()
    print()
    
    # 可视化音程提取过程
    if chroma_c.shape[1] > 0:
        print("=== 可视化音程提取 ===")
        visualize_interval_extraction(chroma_c)
    
    # 创建模型
    model = MusicVersionMatchingModel()
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    print("\n=== 代码组合完成 ===")
    print("你现在可以使用以下功能:")
    print("1. extract_chroma_features() - 提取chroma特征")
    print("2. create_interval_embedding() - 创建音程嵌入")
    print("3. MusicVersionMatchingModel - 版本匹配模型")
    print("4. train_model() - 训练模型")
    print("5. find_matching_versions() - 查找匹配版本")

if __name__ == "__main__":
    main()
