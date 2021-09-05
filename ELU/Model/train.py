from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import time
import torch
import torch.utils.tensorboard as tb
from tqdm.notebook import tqdm

from torchvision import transforms

TRAIN_PATH = "./data/train"
VALID_PATH = "./data/valid"

def train(args):
    from os import path
    activation_func = torch.nn.ReLU
    if args.activation_func == "LReLU":
        activation_func = torch.nn.LeakyReLU
    elif args.activation_func == "ELU":
        activation_func = torch.nn.ELU
    elif args.activation_func == "Mish":
        activation_func = torch.nn.Mish
    elif args.activation_func == "GELU":
        activation_func = torch.nn.GELU
    model = CNNClassifier(activation_function=activation_func, batch_norm=args.batch_norm)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    # Set hyperparameters from the parser
    lr = args.lr
    epochs = args.epoch
    batch_size = args.batchsize
    num_workers = args.num_workers
    # Set up the cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up loss function and optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay = 0.0005, momentum=0.9)
    # optim = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay = 0.0005)

    # Set up training data and validation data
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomHorizontalFlip()
    ])
    data_train = load_data(TRAIN_PATH, transform, num_workers, batch_size)
    data_val = load_data(VALID_PATH, None, num_workers, batch_size)

    # Set up loggers
    log_dir = args.log_dir
    log_time = '{}'.format(time.strftime('%H-%M-%S'))
    log_name = 'SLT10_lr=%s_epoch=%s_batch_size=%s' % (lr, epochs, batch_size)
    logger = tb.SummaryWriter()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train') + '/%s_%s' % (log_name, log_time))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test') + '/%s_%s' % (log_name, log_time))
    global_step = 0

    # Wrap in a progress bar.
    lmbda = lambda epoch: 0.95
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optim, lr_lambda=lmbda)
    start = time.time()
    for epoch in range(epochs):
        # Set the model to training mode.
        model.train()

        train_accuracy_val = list()
        for x, y in data_train:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            train_accuracy_val.append(accuracy(y_pred, y))

            # Compute loss and update model weights.
            loss = loss_func(y_pred, y)

            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Add loss to TensorBoard.
            train_logger.add_scalar('Loss', loss.item(), global_step=global_step)
            global_step += 1

        train_accuracy_total = torch.FloatTensor(train_accuracy_val).mean().item()
        train_logger.add_scalar('Train Accuracy', train_accuracy_total, global_step=global_step)

        # Set the model to eval mode and compute accuracy.
        # No need to change this, but feel free to implement additional logging.
        model.eval()

        accuracys_val = list()

        for x, y in data_val:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            accuracys_val.append(accuracy(y_pred, y))

        accuracy_total = torch.FloatTensor(accuracys_val).mean().item()
        valid_logger.add_scalar('Validation Accuracy', accuracy_total, global_step=global_step)

        scheduler.step()
    print(args.activation_func, args.batch_norm, accuracy_total, time.time() - start, sep=",")
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batchsize', type=int, default=64)
    parser.add_argument('-w', '--num_workers', type=int, default=0)
    parser.add_argument('-l', '--lr', type=float, default=1e-3)
    parser.add_argument('-n', "--batch_norm", default=False, action="store_true")
    parser.add_argument('-a', "--activation_func", type=str, default="ReLU")


    args = parser.parse_args()
    train(args)