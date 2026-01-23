import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # 1. 파일 경로 (File Paths)
    TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "dpr_train_data", "dpr_train.json")
    VAL_DATA_PATH = os.path.join(BASE_DIR, "data", "dpr_train_data", "dpr_dev.json")
    SAVE_DIR = os.path.join(BASE_DIR, "saved_dpr_models", "dpr_finetuned")
    LOSS_HISTORY_FILE = os.path.join(SAVE_DIR, "loss_history.json")
    RESUME_CHECKPOINT_PATH = None 

    # 2. 모델 설정
    Q_ENCODER_ID = 'facebook/dpr-question_encoder-multiset-base'
    CTX_ENCODER_ID = 'facebook/dpr-ctx_encoder-multiset-base'
    
    # 3. 하이퍼파라미터 (과적합 방지 추천 조합 반영)
    BATCH_SIZE = 8 
    ACCUMULATION_STEPS = 4      # [추가] 논리적 배치 크기를 32(8x4)로 확장
    EPOCHS = 10                 # [변경] 충분한 수렴을 위해 10 에포크로 확장
    LEARNING_RATE = 1e-5        # [변경] 안정적인 수렴을 위해 하향 조정
    WEIGHT_DECAY = 0.1          # [추가] 강한 가중치 감쇄로 과적합 방지
    MAX_GRAD_NORM = 0.5         # [추가] 그래디언트 폭주 방지를 위한 클리핑 임계값
    
    MAX_LENGTH = 256
    WARMUP_STEPS = 500          # [변경] 초기 학습 안정화를 위해 500 step 설정
    SEED = 42
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0