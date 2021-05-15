
# def schedule(mile_stone, lr, warmup=False, warmup_epoch=5): # milestone: list of milestones, lr: list of lrs

#     if warmup:
#         step = lr[0] / warmup_epoch
#         next_station = mile_stone[0]
#         def schedule(epoch, lr):
#             if epoch < warmup_epoch and epoch >= 0: # epoch indexed from 0
#                 return  step * (epoch + 1)
#             if epoch <= 80:
#                 return 0.8
#             elif 80 < epoch <= 121:
#                 return 0.08
#             return 0.008
    