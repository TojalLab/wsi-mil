
from fastai.torch_core import show_image, show_images, show_image_batch

def figure_to_numpy(fig):
    import io
    import numpy as np
    # https://stackoverflow.com/a/61443397
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr.transpose([2,0,1]) # CHW

def figure_to_PIL(fig):
    import io
    import PIL.Image
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=fig.dpi)
    return PIL.Image.frombuffer('RGBA', data=io_buf.getvalue(), size=fig.canvas.get_width_height())

def pil_concat_side_by_side(fig1, fig2):
    import PIL.Image
    max_height = max(fig1.height, fig2.height)
    dst = PIL.Image.new('RGB', (fig1.width + fig2.width, max_height))
    dst.paste(fig1, (0,0))
    dst.paste(fig2, (fig1.width, 0))
    return dst

def roc_curves_figure(fpr, tpr, labels=None, cmap='Set1', figsize=(6,6)):
    import matplotlib.pyplot as plt
    from sklearn.metrics import RocCurveDisplay, auc
    ns = len(fpr)
    colors = plt.get_cmap(cmap).colors
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    fig.set_facecolor('white')
    if labels is None:
        labels = list(range(ns))
    for i in range(ns):
        area = auc(fpr[i], tpr[i])
        display = RocCurveDisplay(
            fpr=fpr[i],
            tpr=tpr[i]
        )
        display.plot(ax=ax, name=f"AUC [{labels[i]}] = {area:.2f}", color=colors[i])
    plt.close()
    return fig

def pr_curves_figure(prec, recall, labels=None, cmap='Set1', figsize=(6,6)):
    import matplotlib.pyplot as plt
    from sklearn.metrics import PrecisionRecallDisplay, auc
    ns = len(prec)
    colors = plt.get_cmap(cmap).colors
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    fig.set_facecolor('white')
    if labels is None:
        labels = list(range(ns))
    for i in range(ns):
        area = auc(recall[i], prec[i])
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=prec[i]
        )
        display.plot(ax=ax, name=f"AUC [{labels[i]}] = {area:.2f}", color=colors[i])
    plt.close()
    return fig

def confusion_matrix_figure(cm, labels=None, figsize=(6,6)):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    fig.set_facecolor('white')
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax,xticks_rotation='vertical')
    plt.close()
    return fig
