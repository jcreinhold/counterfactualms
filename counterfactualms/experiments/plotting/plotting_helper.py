from collections import OrderedDict
from functools import partial
import inspect
import os
import re
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import pyro
import torch

warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_grad_enabled(False)

mpl.rcParams['figure.dpi'] = 300

img_cm = 'gray'
diff_cm = 'seismic'

from counterfactualms.datasets.calabresi import CalabresiDataset

resize = None
crop_size = None
calabresi_test = None
n_rot90 = 0

from counterfactualms.experiments import calabresi  # noqa: F401
from counterfactualms.experiments.calabresi.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401

models = {}
loaded_models = {}
device = None

variables = (
    'sex',
    'age',
    'brain_volume',
    'ventricle_volume',
    'lesion_volume',
    'duration',
    'edss',
)
var_name = {
    'ventricle_volume': 'v',
    'brain_volume': 'b',
    'lesion_volume': 'l',
    'sex': 's',
    'age': 'a',
    'edss': 'e',
    'duration': 'd',
}
value_fmt = {
    'edss': lambda s: rf'{np.round(s,2):.2g}',
    'duration': lambda s: rf'{np.round(s,2):.2g}\,\mathrm{{y}}',
    'ventricle_volume': lambda s: rf'{np.round(s/1000,2):.2g}\,\mathrm{{ml}}',
    'brain_volume': lambda s: rf'{int(np.round(s/1000)):d}\,\mathrm{{ml}}',
    'lesion_volume': lambda s: rf'{np.round(s/1000,2):.2g}\,\mathrm{{ml}}',
    'age': lambda s: rf'{int(s):d}\,\mathrm{{y}}',
    'sex': lambda s: r'{}'.format(['\mathrm{F}', '\mathrm{M}'][int(s)]),
    'type': lambda s: r'{}'.format(['\mathrm{HC}', '\mathrm{MS}'][int(s)]),
}
save_fmt = {
    'edss': lambda s: f'{np.round(s,2):.2g}',
    'duration': lambda s: f'{np.round(s,2):.2g}',
    'ventricle_volume': lambda s: f'{np.round(s/1000,2):.2g}',
    'brain_volume': lambda s: f'{int(np.round(s/1000)):d}',
    'lesion_volume': lambda s: f'{np.round(s/1000,2):.2g}',
    'age': lambda s: f'{int(s):d}',
    'sex': lambda s: '{}'.format(['F', 'M'][int(s)]),
    'type': lambda s: '{}'.format(['HC', 'MS'][int(s)]),
}

imshow_kwargs = dict(vmin=0., vmax=255.)
_buffers_to_load = ('norm', 'permutation')


def get_best_model(model_paths):
    min_score = np.inf
    min_klz = np.inf
    idx = None
    model_paths = [mp for mp in model_paths if 'last' not in mp]
    for i, mp in enumerate(model_paths):
        ms = mp.split('=')[-2:]
        ms = [re.sub('[^\d.-]+','', m) for m in ms]
        klz = float(ms[0][:-1])
        score = float(ms[1][:-1])
        if score < min_score:
            min_score = score
            min_klz = klz
            idx = i
        elif score == min_score:
            if klz < min_klz:
                min_klz = klz
                idx = i
    return model_paths[idx]


def setup(model_path, csv_path, exp_crop_size=(224, 224), exp_resize=(128,128), use_gpu=True):
    """ run this first with paths to models corresponding to experiments """
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    hparams = ckpt['hyper_parameters']
    exp = hparams['model']
    model_class = MODEL_REGISTRY[exp]
    model_params = {
        k: v for k, v in hparams.items() if (k in inspect.signature(model_class.__init__).parameters
                                             or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
                                             or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters)
    }
    model_params['img_shape'] = hparams['resize'] if 'resize' in hparams else exp_resize
    new_state_dict = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        new_key = key.replace('pyro_model.', '')
        new_state_dict[new_key] = value
    loaded_model = model_class(**model_params)
    loaded_model.load_state_dict(new_state_dict)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    for p in loaded_model._buffers.keys():
        if any([(b in p) for b in _buffers_to_load]):
            setattr(loaded_model, p, getattr(loaded_model, p).to(device))
    loaded_model.eval()
    global loaded_models
    loaded_models[exp] = loaded_model.to(device)
    global crop_size
    crop_size = exp_crop_size
    global resize
    resize = exp_resize
    global calabresi_test
    calabresi_test = CalabresiDataset(csv_path, crop_type='center', resize=resize, crop_size=crop_size)
    def sample_pgm(num_samples, model):
        with pyro.plate('observations', num_samples):
            return model.pgm_model()
    global models
    models[exp] = partial(sample_pgm, model=loaded_model)


def fmt_intervention(intervention):
    if isinstance(intervention, str):
        var, value = intervention[3:-1].split('=')
        return f"$do({var_name[var]}={value_fmt[var](value)})$"
    else:
        all_interventions = ',\n'.join([f'${var_name[k]}={value_fmt[k](v)}$' for k, v in intervention.items()])
        return f"do({all_interventions})"


def fmt_save(intervention):
    if isinstance(intervention, str):
        var, value = intervention[3:-1].split('=')
        return f"do({var_name[var]}={save_fmt[var](value)})"
    else:
        all_interventions = ','.join([f'{var_name[k]}={save_fmt[k](v)}' for k, v in intervention.items()])
        return f"do({all_interventions})"


def prep_data(batch):
    x = 255. * batch['image'].float().unsqueeze(0)
    age = batch['age'].unsqueeze(0).unsqueeze(0).float()
    sex = batch['sex'].unsqueeze(0).unsqueeze(0).float()
    ventricle_volume = batch['ventricle_volume'].unsqueeze(0).unsqueeze(0).float()
    brain_volume = batch['brain_volume'].unsqueeze(0).unsqueeze(0).float()
    lesion_volume = batch['lesion_volume'].unsqueeze(0).unsqueeze(0).float()
    edss = batch['edss'].unsqueeze(0).unsqueeze(0).float()
    duration = batch['duration'].unsqueeze(0).unsqueeze(0).float()
    type = batch['type'].unsqueeze(0).unsqueeze(0).float()
    slice_number = batch['slice_number'].unsqueeze(0).unsqueeze(0).float()
    return {'x': x, 'age': age, 'sex': sex, 'ventricle_volume': ventricle_volume,
            'brain_volume': brain_volume, 'lesion_volume': lesion_volume,
            'edss': edss, 'duration': duration, 'type': type, 'slice_number': slice_number}


def plot_gen_intervention_range(model_name, interventions, idx, normalise_all=True, num_samples=32):
    fig, ax = plt.subplots(3, len(interventions), figsize=(1.6 * len(interventions), 5), gridspec_kw=dict(wspace=0, hspace=0))
    lim = 0
    orig_data = prep_data(calabresi_test[idx])
    ms_type = orig_data['type']
    del orig_data['type']
    og = {k: v.to(device) for k, v in orig_data.items()}
    imgs = []
    for intervention in interventions:
        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]).to(device) for k, v in intervention.items()}
        counterfactual = loaded_models[model_name].counterfactual(og, cond, num_samples)
        counterfactual = {k: v.detach().cpu() for k, v in counterfactual.items()}
        imgs += [counterfactual['x']]
        diff = (imgs[-1] - orig_data['x']).squeeze()
        lim = np.maximum(lim, diff.abs().max())

    if 'cuda' in str(device):
        del og, cond
        torch.cuda.empty_cache()

    for i, intervention in enumerate(interventions):
        x = imgs[i]
        x_test = orig_data['x']
        diff = (x - x_test).squeeze()
        if not normalise_all:
            lim = diff.abs().max()
        ax[0, i].imshow(np.rot90(x_test.squeeze(), n_rot90), img_cm, **imshow_kwargs)
        ax[0, i].set_title(fmt_intervention(intervention))
        ax[1, i].imshow(np.rot90(x.squeeze(), n_rot90), img_cm, **imshow_kwargs)
        ax[2, i].imshow(np.rot90(diff, n_rot90), diff_cm, clim=[-lim, lim])
        for axi in ax[:, i]:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())

    orig_data['type'] = ms_type
    suptitle = ('$s={sex}$; $a={age}$; $b={brain_volume}$; $v={ventricle_volume}$; $l={lesion_volume}$; $d={duration}$; $e={edss}$').format(
        **{att: value_fmt[att](orig_data[att].item()) for att in variables}
    )
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    plt.show()


def interactive_plot(model_name):
    def _to_png(x):
        if hasattr(x, 'numpy'):
            x = x.numpy()
        x = np.rot90(x.squeeze(), n_rot90)
        x = np.clip(x, 0., 255.)
        x = x.astype(np.uint8)
        x = Image.fromarray(x)
        x = x.resize(resize, resample=Image.BILINEAR)
        return x

    def plot_intervention(intervention, idx, num_samples=32, save_image_dir='', save=False):
        fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), gridspec_kw=dict(wspace=0, hspace=0))
        orig_data = prep_data(calabresi_test[idx])
        ms_type = orig_data['type']
        del orig_data['type']
        x_test = orig_data['x']
        og = {k: v.to(device) for k, v in orig_data.items()}
        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]).to(device) for k, v in intervention.items()}
        counterfactual = loaded_models[model_name].counterfactual(og, cond, num_samples)
        counterfactual = {k: v.detach().cpu() for k, v in counterfactual.items()}
        x = counterfactual['x']
        diff = (x - x_test).squeeze()
        lim = diff.abs().max()
        ax[1].set_title('Original')
        ax[1].imshow(np.rot90(x_test.squeeze(), n_rot90), img_cm, **imshow_kwargs)
        ax[2].set_title(fmt_intervention(intervention))
        ax[2].imshow(np.rot90(x.squeeze(), n_rot90), img_cm, **imshow_kwargs)
        ax[3].set_title('Difference')
        im = ax[3].imshow(np.rot90(diff, n_rot90), diff_cm, clim=[-lim, lim])
        plt.colorbar(im)
        for axi in ax:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())

        if 'cuda' in str(device):
            del og, cond
            torch.cuda.empty_cache()

        orig_data['type'] = ms_type
        att_str = ('$s={sex}$\n$a={age}$\n$b={brain_volume}$\n$v={ventricle_volume}$\n$l={lesion_volume}$\n'
                   '$d={duration}$\n$e={edss}$\n$t={type}$').format(
            **{att: value_fmt[att](orig_data[att].item()) for att in variables + ('type',)}
        )

        ax[0].text(0.5, 0.5, att_str, horizontalalignment='center',
                      verticalalignment='center', transform=ax[0].transAxes,
                      fontsize=mpl.rcParams['axes.titlesize'])
        fig.tight_layout()
        if save_image_dir and save:
            x = _to_png(x)
            x_test = _to_png(x_test)
            intervention_str = fmt_save(intervention)
            x.save(os.path.join(save_image_dir, intervention_str+'_cf.png'))
            x_test.save(os.path.join(save_image_dir, intervention_str+'_orig.png'))
            plt.savefig(intervention_str+'_full.pdf');
        plt.show()

    from ipywidgets import (
        interactive, IntSlider, FloatSlider, HBox, VBox,
        Checkbox, Dropdown, Text, Button, BoundedIntText
    )
    from IPython.display import display

    def plot(image, age, sex, brain_volume, ventricle_volume, lesion_volume,
             duration, edss,
             do_age, do_sex, do_brain_volume, do_ventricle_volume, do_lesion_volume,
             do_duration, do_edss,
             save_image_dir, save):
        intervention = {}
        if do_age:
            intervention['age'] = age
        if do_sex:
            intervention['sex'] = sex
        if do_brain_volume:
            intervention['brain_volume'] = brain_volume * 1000.
        if do_ventricle_volume:
            intervention['ventricle_volume'] = ventricle_volume * 1000.
        if do_lesion_volume:
            intervention['lesion_volume'] = lesion_volume * 1000.
        if do_duration:
            intervention['duration'] = duration
        if do_edss:
            intervention['edss'] = edss

        plot_intervention(intervention, image, save_image_dir=save_image_dir, save=save)

    w = interactive(plot, image=BoundedIntText(min=0, max=len(calabresi_test)-1, description='Image #'),
        age=FloatSlider(min=20., max=80., step=1., continuous_update=False, description='Age'),
        do_age=Checkbox(description='do(age)'),
        sex=Dropdown(options=[('female', 0.), ('male', 1.)], description='Sex'),
        do_sex=Checkbox(description='do(sex)'),
        brain_volume=FloatSlider(min=600., max=1800., step=0.5, continuous_update=False, description='Brain Volume (ml):', style={'description_width': 'initial'}),
        do_brain_volume=Checkbox(description='do(brain_volume)'),
        ventricle_volume=FloatSlider(min=1e-5, max=88., step=0.1, continuous_update=False, description='Ventricle Volume (ml):', style={'description_width': 'initial'}),
        do_ventricle_volume=Checkbox(description='do(ventricle_volume)'),
        lesion_volume=FloatSlider(min=1e-5, max=66., step=0.1, continuous_update=False, description='Lesion Volume (ml):', style={'description_width': 'initial'}),
        do_lesion_volume=Checkbox(description='do(lesion_volume)'),
        duration=FloatSlider(min=1e-5, max=24., step=1., continuous_update=False, description='Duration (y):', style={'description_width': 'initial'}),
        do_duration=Checkbox(description='do(duration)'),
        edss=FloatSlider(min=1e-5, max=10., step=1., continuous_update=False, description='EDSS:', style={'description_width': 'initial'}),
        do_edss=Checkbox(description='do(edss)'),
        save_image_dir=Text(value='', placeholder='Full path', description='Save Image Directory:', style={'description_width': 'initial'}),
        save=Checkbox(description='Save')
        )

    n = len(variables)
    ui = VBox([HBox([w.children[0], w.children[-3], w.children[-2]]),  # image # and save_image_dir
               VBox([HBox([w.children[i], w.children[i+n]]) for i in range(1,n+1)]), # vars and intervention checkboxes
               w.children[-1]])  # show image
    display(ui)
    w.update()
