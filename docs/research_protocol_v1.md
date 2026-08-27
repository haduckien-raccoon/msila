# RQ1 - Adapter residual
- So sánh: Frozen DINOv3 + decoder và DINOv3 + residual adapter + decoder
- Tiêu chí: cải thiện macro AU-Pro_0.05 và tốt hơn ít nhất 6/8 category
# RQ2 - Đa tỷ lệ
- Chuỗi avlation: Deep/local --> Multi-;ayer --> Local+ Context --> Attention Fusion
- Tiee chí: tăng điểm tổng thể hoặc tăng điểm trên nhóm tiny-defect
# RQ3 - Illumination
- So sánh: Augmentation vs Augmentation + L_illum
- Theo dõi đồng thời: Score_priv, Score_mix, Gap_illum = Score_priv - Score_mix
# RQ4 - Synthetic anomaly
- So sánh: Random mask vs Boundary/Tiny-aware
- Theo dõi SegF1 vs Boundary-F1