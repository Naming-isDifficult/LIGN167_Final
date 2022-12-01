from WaveNet.Trainer import Trainer

if __name__ == '__main__':
    
    model_trainer = Trainer()
    model_trainer.initialize_model(batch_size=4,\
                                   stack_size=4, layer_per_stack=12,\
                                   input_dim=1, res_dim=32, output_dim=256)
                                   #receptive_field=16384
                                   #a little bit longer than 1s
    print('receptive_field:', model_trainer.wavenet_model.get_receptive_field())
    model_trainer.train_model()