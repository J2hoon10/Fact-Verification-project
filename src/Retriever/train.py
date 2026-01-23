import os
import torch
import json
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

from dpr_architecture import DPRBiEncoder
from dataset_loader import DPRDataset, collate_fn
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, scheduler, epoch, loss_history, best_val_loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': loss_history,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, filepath)
    print(f"\n[Checkpoint] Saved to {filepath}")

def validate(model, dataloader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for q_inputs, p_inputs in tqdm(dataloader, desc="Validating"):
            q_input_ids = q_inputs['input_ids'].to(device)
            q_attn_mask = q_inputs['attention_mask'].to(device)
            p_input_ids = p_inputs['input_ids'].to(device)
            p_attn_mask = p_inputs['attention_mask'].to(device)

            loss = model(q_input_ids, q_attn_mask, p_input_ids, p_attn_mask)
            total_val_loss += loss.item()
    return total_val_loss / len(dataloader)

def train():
    cfg = config.Config
    set_seed(cfg.SEED)
    device = cfg.DEVICE

    # 1. 토크나이저 로드 및 스페셜 토큰 추가
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(cfg.Q_ENCODER_ID)
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(cfg.CTX_ENCODER_ID)

    special_tokens = {"additional_special_tokens": ["[SEP]", "[CTX]"]}
    q_tokenizer.add_special_tokens(special_tokens)
    ctx_tokenizer.add_special_tokens(special_tokens)

    # 2. 모델 초기화 및 임베딩 리사이징
    model = DPRBiEncoder().to(device)
    model.resize_token_embeddings(len(q_tokenizer), len(ctx_tokenizer))

    # 3. 데이터셋 로드
    train_dataset = DPRDataset(cfg.TRAIN_DATA_PATH)
    val_dataset = DPRDataset(cfg.VAL_DATA_PATH)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
        collate_fn=lambda x: collate_fn(x, q_tokenizer, ctx_tokenizer),
        num_workers=cfg.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=lambda x: collate_fn(x, q_tokenizer, ctx_tokenizer),
        num_workers=cfg.NUM_WORKERS
    )

    # 4. 옵티마이저 및 스케줄러 수정
    # [수정] 가중치 감쇄(Weight Decay)를 적용하여 정규화 강화
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # [수정] Accumulation 스텝을 고려하여 전체 최적화 스텝 재계산
    total_optimization_steps = (len(train_loader) // cfg.ACCUMULATION_STEPS) * cfg.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.WARMUP_STEPS, num_training_steps=total_optimization_steps
    )

    # 5. Resume 로직 (기존 유지)
    start_epoch = 0
    best_val_loss = float('inf')
    loss_history = []

    if cfg.RESUME_CHECKPOINT_PATH and os.path.exists(cfg.RESUME_CHECKPOINT_PATH):
        checkpoint = torch.load(cfg.RESUME_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        loss_history = checkpoint.get('loss_history', [])

    # 6. 학습 루프 수정
    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        train_loss_sum = 0
        optimizer.zero_grad() # 루프 밖으로 이동
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")

        for step, (q_inputs, p_inputs) in enumerate(pbar):
            q_ids = q_inputs['input_ids'].to(device)
            q_mask = q_inputs['attention_mask'].to(device)
            p_ids = p_inputs['input_ids'].to(device)
            p_mask = p_inputs['attention_mask'].to(device)

            # Forward
            loss = model(q_ids, q_mask, p_ids, p_mask)
            
            # [수정] Gradient Accumulation 적용
            # 전체 배치의 평균 손실을 유지하기 위해 나눔
            loss = loss / cfg.ACCUMULATION_STEPS
            loss.backward()

            # [수정] 지정된 accumulation 스텝마다 가중치 업데이트
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                # 클리핑 임계값을 0.5로 하향하여 안정성 확보
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 시각화를 위해 원래 스케일의 loss 합산
            train_loss_sum += loss.item() * cfg.ACCUMULATION_STEPS
            pbar.set_postfix({'loss': f"{loss.item() * cfg.ACCUMULATION_STEPS:.4f}"})

        avg_val_loss = validate(model, val_loader, device)
        avg_train_loss = train_loss_sum / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Best Model 저장 (기존 유지)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(cfg.SAVE_DIR, "best_model")
            os.makedirs(save_path, exist_ok=True)
            model.question_encoder.q_model.save_pretrained(f"{save_path}/question_encoder")
            model.ctx_encoder.ctx_model.save_pretrained(f"{save_path}/ctx_encoder")
            q_tokenizer.save_pretrained(f"{save_path}/question_encoder")
            ctx_tokenizer.save_pretrained(f"{save_path}/ctx_encoder")

        save_checkpoint(model, optimizer, scheduler, epoch, loss_history, best_val_loss, f"{cfg.SAVE_DIR}/checkpoint_latest.pt")

if __name__ == "__main__":
    train()