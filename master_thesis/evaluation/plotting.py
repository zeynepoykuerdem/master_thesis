"""
Plotting functions for memorization analysis
a) Training Loss Curves
b) Label Corruption vs Time to Overfit  
c) Label Corruption vs Test Error
d) GradCAM Visualization
e) Integrated Gradients Visualization
"""

import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
mean = np.array([0.4914, 0.4822, 0.4465])
std  = np.array([0.2023, 0.1994, 0.2010])


def plot_history(history, title='', save_path=None):
    loss_key = 'train_loss' if 'train_loss' in history else 'loss'
    acc_key  = 'train_acc'  if 'train_acc'  in history else 'acc'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if loss_key in history:
        ax1.plot(history[loss_key], label='Train Loss', color='blue')
    if 'test_loss' in history:
        ax1.plot(history['test_loss'], label='Test Loss', color='red')
    ax1.set_title(f'{title} Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    if acc_key in history:
        ax2.plot(history[acc_key], label='Train Accuracy', color='green')
    if 'test_acc' in history:
        ax2.plot(history['test_acc'], label='Test Accuracy', color='orange')
    ax2.set_title(f'{title} Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_loss_curves(results_corruption, exp_names, save_path=None):
    """(a) Training loss curves for experimental settings"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.viridis(np.linspace(0, 1, len(exp_names)))
    

    for i,exp_name in enumerate(exp_names):
        ax.plot(
            results_corruption[exp_name]['loss_curve'],
            label=f'{exp_name.replace("_", " ").title()}',
            color=colors[i],
        )

    ax.set_title('(a) Training Loss Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# Overfitting Graph : For each corruption ratio, find the epoch where training loss < 0.1 and plot vs corruption ratio for CNN and ResNet56
def plot_label_corruption_vs_time_to_overfit(results_cnn,results_resnet, corruption_ratios, save_path=None):
    """(b) Label corruption ratio vs epochs needed to overfit- Simple CNN vs Resnet56"""
    cnnoverfit_times = []
    resnetoverfit_times = []
    for ratio in corruption_ratios:
        cnn_loss_curve    = results_cnn[ratio]['loss_history']
        resnet_loss_curve = results_resnet[ratio]['loss_history']
        cnn_overfit_epoch = next(
            (i for i, l in enumerate(cnn_loss_curve) if l < 0.1),
            len(cnn_loss_curve)
        )
        resnet_overfit_epoch = next(
            (i for i, l in enumerate(resnet_loss_curve) if l < 0.1),
            len(resnet_loss_curve)
        )
        cnnoverfit_times.append(cnn_overfit_epoch)
        resnetoverfit_times.append(resnet_overfit_epoch)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r * 100 for r in corruption_ratios], cnnoverfit_times, 'bo-', label='CNN'
        )
    ax.plot(
        [r * 100 for r in corruption_ratios], resnetoverfit_times, 'ro-', label='ResNet56'
        )
    
    ax.set_title('(b) Label Corruption vs Time to Overfit')
    ax.set_xlabel('Label Corruption (%)')
    ax.set_ylabel('Epochs to Overfit')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_label_corruption_vs_test_error(results_cnn,results_resnet, corruption_ratios, save_path=None):
    """(c) Label corruption ratio vs test error"""
    cnn_test_errors = [
        100 - results_cnn[r]['test_acc'] for r in corruption_ratios
    ]
    resnet_test_errors = [
        100 - results_resnet[r]['test_acc'] for r in corruption_ratios
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [r * 100 for r in corruption_ratios], cnn_test_errors, 'bo-', label='CNN'
        )

    ax.plot(
        [r * 100 for r in corruption_ratios], resnet_test_errors, 'ro-', label='ResNet56'
        )
    ax.axhline(y=90, color='gray', linestyle='--', label='Chance Level (~90%)')
    ax.set_title('(c) Test Error vs Label Corruption')
    ax.set_xlabel('Label Corruption (%)')
    ax.set_ylabel('Test Error (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def gradcam_visualization(images, labels, heatmaps, predictions,  title='GradCAM', save_path=None, n=5):
    """GradCAM heatmap visualization"""
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    fig.suptitle(title, fontsize=14)

    for i in range(n):
        img_show = images[i].cpu().numpy().transpose(1, 2, 0)
        img_show = std * img_show + mean
        img_show = np.clip(img_show, 0, 1)

        axes[0, i].imshow(img_show)
        axes[0, i].set_title(
            f'True: {CLASS_NAMES[labels[i]]}\n'
            f'Pred: {CLASS_NAMES[predictions[i]]}',
            fontsize=8
        )
        axes[0, i].axis('off')

        axes[1, i].imshow(heatmaps[i], cmap='jet')
        axes[1, i].axis('off')

        axes[2, i].imshow(img_show)
        axes[2, i].imshow(heatmaps[i], cmap='jet', alpha=0.5)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Original', size=10)
    axes[1, 0].set_ylabel('GradCAM',  size=10)
    axes[2, 0].set_ylabel('Overlay',  size=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def integrated_gradients_visualization(images, labels, attributions,
                                        title='Integrated Gradients',
                                        save_path=None, n=5):
    """Integrated Gradients attribution visualization"""
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    fig.suptitle(title, fontsize=14)

    for i in range(n):
        img_show = images[i].cpu().numpy().transpose(1, 2, 0)
        img_show = std * img_show + mean
        img_show = np.clip(img_show, 0, 1)

        attr = attributions[i]
        attr = np.abs(attr).sum(axis=0)
        if attr.max() > 0:
            attr = attr / attr.max()

        axes[0, i].imshow(img_show)
        axes[0, i].set_title(CLASS_NAMES[labels[i]], fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(attr, cmap='hot')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original',   size=10)
    axes[1, 0].set_ylabel('Attribution', size=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

