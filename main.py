import util

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GAN')
    parser.add_argument('--set', type=str, help="Accepted values: {mnist, cifar10}")
    parser.add_argument('--train', default=False, action='store_true', help='If set the training procedure is done')
    args = parser.parse_args()

    if args.set == 'mnist':
        import mnist as gan
        output = './resources/mnist/'
    elif args.set == 'cifar10':
        import cifar_10 as gan
        output = './resources/cifar_10/'
    else:
        print('Invalid data set')
        exit(1)

    checkpoint_prefix = output + 'ckpt'
    util.generate(output, gan, output, checkpoint_prefix, args.train)
