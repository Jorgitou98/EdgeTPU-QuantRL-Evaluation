import argparse
import subprocess
import os
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
        '-m', '--model', required=True, type=int, choices=[i for i in range(1,9)], help='Integer indicating the model to create. It must be an integer value between 1 and 6. Run python train_model.available_models() to see the description of available models.'
)
parser.add_argument(
        '-i', '--iters', required = True, type=int, help= 'Number of training iters to run'
)
parser.add_argument(
        '-d', '--dim', required = True, type=int, help= 'Input dimension'
)
args = parser.parse_args()


train = 'python train_ppo.py --model={} --gpu=gpu0 --workers=1 --save-name=model{} --iters={}'.format(args.model, args.model, args.iters)
process = subprocess.Popen(train.split())
output, error = process.communicate()

export = 'python model_saver.py checkpoints/ppo/model{}/checkpoint_{}/checkpoint-{} exported_models/checkpoint-{}-model{}'.format(args.model, '0'*(6-len(str(args.iters)))+str(args.iters), args.iters, args.iters, args.model)
process = subprocess.Popen(export.split())
output, error = process.communicate()

convert = 'python tflite_converter_pb.py exported_models/checkpoint-{}-model{} exported_models/checkpoint-{}-model{}.tflite'.format(args.iters, args.model, args.iters, args.model)
process = subprocess.Popen(convert.split())
output, error = process.communicate()

if not os.path.exists('datasets/dataset-model{}.npy'.format(args.model)):
  create_dataset = 'python dataset_creator.py {} datasets/dataset-model{}'.format(args.dim, args.model)
  process = subprocess.Popen(create_dataset.split())
  output, error = process.communicate()


quant8b = 'python quantizer8b.py datasets/dataset-model{}.npy exported_models/checkpoint-{}-model{} exported_models/checkpoint-{}-model{}-quant8.tflite'.format(args.model, args.iters, args.model, args.iters, args.model)
process = subprocess.Popen(quant8b.split())
output, error = process.communicate()


quant16b = 'python quantizer16b.py exported_models/checkpoint-{}-model{} exported_models/checkpoint-{}-model{}-quant16.tflite'.format(args.iters, args.model, args.iters, args.model)
process = subprocess.Popen(quant16b.split())
output, error = process.communicate()

edge_tpu = 'edgetpu_compiler exported_models/checkpoint-{}-model{}-quant8.tflite'.format(args.iters, args.model)
process = subprocess.Popen(edge_tpu.split())
output, error = process.communicate()

move_tpu_files = 'mv checkpoint-{}-model{}-quant8_edgetpu.tflite exported_models/checkpoint-{}-model{}-quant8_edgetpu.tflite'.format(args.iters, args.model, args.iters, args.model)
process = subprocess.Popen(move_tpu_files.split())
output, error = process.communicate()
