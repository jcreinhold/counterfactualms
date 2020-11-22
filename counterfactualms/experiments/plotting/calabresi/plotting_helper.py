from collections import OrderedDict
from functools import partial
import inspect
import os
import traceback
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pyro
import torch

warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_grad_enabled(False)

mpl.rcParams['figure.dpi'] = 300

img_cm = 'Greys_r'
diff_cm = 'seismic'

from counterfactualms.datasets.medical.calabresi import CalabresiDataset

csv = "/iacl/pg20/jacobr/calabresi/png/csv/test_png.csv"
downsample = 2
crop_size = (224, 224)
calabresi_test = CalabresiDataset(csv, crop_type='center', downsample=downsample, crop_size=crop_size)
n_rot90 = 0

from counterfactualms.experiments.medical import calabresi  # noqa: F401
from counterfactualms.experiments.medical.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401

experiments = ['ConditionalVISEM']
models = {}
loaded_models = {}

variables = (
    'sex',
    'age',
    'brain_volume',
    'ventricle_volume',
    'lesion_volume',
    'duration',
    'score',
    'slice_number',
)
var_name = {
    'ventricle_volume': 'v',
    'brain_volume': 'b',
    'lesion_volume': 'l',
    'sex': 's',
    'age': 'a',
    'score': 'e',
    'duration': 'd',
    'slice_number': 'n',
}
value_fmt = {
    'score': lambda s: rf'{float(s):.3g}',
    'duration': lambda s: rf'{float(s):.3g}\,\mathrm{{y}}',
    'ventricle_volume': lambda s: rf'{float(s)/1000:.3g}\,\mathrm{{ml}}',
    'brain_volume': lambda s: rf'{float(s)/1000:.3g}\,\mathrm{{ml}}',
    'lesion_volume': lambda s: rf'{float(s)/1000:.3g}\,\mathrm{{ml}}',
    'age': lambda s: rf'{int(s):d}\,\mathrm{{y}}',
    'sex': lambda s: r'{}'.format(['\mathrm{F}', '\mathrm{M}'][int(s)]),
    'type': lambda s: r'{}'.format(['\mathrm{HC}', '\mathrm{MS}'][int(s)]),
    'slice_number': lambda s: rf'{int(s):d}',
}

def setup(model_paths):
    """ run this first with paths to models corresponding to experiments """
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    if len(model_paths) != len(experiments):
        raise ValueError('Provided paths do not match number of experiments')
    for exp, path in zip(experiments, model_paths):
        try:
            ckpt = torch.load(path, map_location=torch.device('cpu'))
            hparams = ckpt['hyper_parameters']
            model_class = MODEL_REGISTRY[hparams['model']]
            model_params = {
                k: v for k, v in hparams.items() if (k in inspect.signature(model_class.__init__).parameters
                                                     or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
                                                     or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters)
            }
            model_params['img_shape'] = hparams['crop_size']
            new_state_dict = OrderedDict()
            for key, value in ckpt['state_dict'].items():
                new_key = key.replace('pyro_model.', '')
                new_state_dict[new_key] = value
            loaded_model = model_class(**model_params)
            loaded_model.load_state_dict(new_state_dict)
            for p in loaded_model._buffers.keys():
                if 'norm' in p:
                    setattr(loaded_model, p, getattr(loaded_model, p))
            loaded_model.eval()
            global loaded_models
            loaded_models[exp] = loaded_model
            def sample_pgm(num_samples, model):
                with pyro.plate('observations', num_samples):
                    return model.pgm_model()
            global models
            models[exp] = partial(sample_pgm, model=loaded_model)
        except Exception as e:
            print(e)
            traceback.print_exc()


def fmt_intervention(intervention):
    if isinstance(intervention, str):
        var, value = intervention[3:-1].split('=')
        return f"$do({var_name[var]}={value_fmt[var](value)})$"
    else:
        all_interventions = ',\n'.join([f'${var_name[k]}={value_fmt[k](v)}$' for k, v in intervention.items()])
        return f"do({all_interventions})"


def prep_data(batch):
    x = batch['image'].unsqueeze(0) * 255.
    age = batch['age'].unsqueeze(0).unsqueeze(0).float()
    sex = batch['sex'].unsqueeze(0).unsqueeze(0).float()
    ventricle_volume = batch['ventricle_volume'].unsqueeze(0).unsqueeze(0).float()
    brain_volume = batch['brain_volume'].unsqueeze(0).unsqueeze(0).float()
    lesion_volume = batch['lesion_volume'].unsqueeze(0).unsqueeze(0).float()
    total_ventricle_volume = batch['total_ventricle_volume'].unsqueeze(0).unsqueeze(0).float()
    total_brain_volume = batch['total_brain_volume'].unsqueeze(0).unsqueeze(0).float()
    total_lesion_volume = batch['total_lesion_volume'].unsqueeze(0).unsqueeze(0).float()
    score = batch['score'].unsqueeze(0).unsqueeze(0).float()
    duration = batch['duration'].unsqueeze(0).unsqueeze(0).float()
    type = batch['type'].unsqueeze(0).unsqueeze(0).float()
    slice_number = batch['slice_number'].unsqueeze(0).unsqueeze(0).float()
    x = x.float()
    return {'x': x, 'age': age, 'sex': sex, 'ventricle_volume': ventricle_volume,
            'brain_volume': brain_volume, 'lesion_volume': lesion_volume,
            'total_ventricle_volume': total_ventricle_volume,
            'total_brain_volume': total_brain_volume,
            'total_lesion_volume': total_lesion_volume,
            'score': score, 'duration': duration, 'type': type,
            'slice_number': slice_number}


def plot_gen_intervention_range(model_name, interventions, idx, normalise_all=True, num_samples=32):
    fig, ax = plt.subplots(3, len(interventions), figsize=(1.6 * len(interventions), 5), gridspec_kw=dict(wspace=0, hspace=0))
    lim = 0
    orig_data = prep_data(calabresi_test[idx])
    ms_type = orig_data['type']
    del orig_data['type']
    imgs = []
    for intervention in interventions:
        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]) for k, v in intervention.items()}
        counterfactual = loaded_models[model_name].counterfactual(orig_data, cond, num_samples)
        imgs += [counterfactual['x']]
        diff = (orig_data['x'] - imgs[-1]).squeeze()
        lim = np.maximum(lim, diff.abs().max())

    for i, intervention in enumerate(interventions):
        x = imgs[i]
        x_test = orig_data['x']
        diff = (x_test - x).squeeze()
        if not normalise_all:
            lim = diff.abs().max()
        ax[0, i].imshow(np.rot90(x_test.squeeze(), n_rot90), img_cm, vmin=0, vmax=255)
        ax[0, i].set_title(fmt_intervention(intervention))
        ax[1, i].imshow(np.rot90(x.squeeze(), n_rot90), img_cm, vmin=0, vmax=255)
        ax[2, i].imshow(np.rot90(diff, n_rot90), diff_cm, clim=[-lim, lim])
        for axi in ax[:, i]:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())

    orig_data['type'] = ms_type
    suptitle = r'$s={sex}; a={age}; b={brain_volume}; v={ventricle_volume}; l={lesion_volume} d={duration}; e={score}; n={slice_number}$'.format(
        **{att: value_fmt[att](orig_data[att].item()) for att in variables}
    )
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()


def interactive_plot(model_name):
    def plot_intervention(intervention, idx, num_samples=32):
        fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), gridspec_kw=dict(wspace=0, hspace=0))
        orig_data = prep_data(calabresi_test[idx])
        ms_type = orig_data['type']
        del orig_data['type']
        x_test = orig_data['x']
        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]) for k, v in intervention.items()}
        counterfactual = loaded_models[model_name].counterfactual(orig_data, cond, num_samples)
        x = counterfactual['x']
        diff = (x_test - x).squeeze()
        lim = diff.abs().max()
        ax[1].set_title('Original')
        ax[1].imshow(np.rot90(x_test.squeeze(), n_rot90), img_cm, vmin=0, vmax=255)
        ax[2].set_title(fmt_intervention(intervention))
        ax[2].imshow(np.rot90(x.squeeze(), n_rot90), img_cm, vmin=0, vmax=255)
        ax[3].set_title('Difference')
        ax[3].imshow(np.rot90(diff, n_rot90), diff_cm, clim=[-lim, lim])
        for axi in ax:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())

        orig_data['type'] = ms_type
        att_str = '$s={sex}$\n$a={age}$\n$b={brain_volume}$\n$v={ventricle_volume}$\n$l={lesion_volume}$\n$d={duration}$\n$e={score}$\n$t={type}$\n$n={slice_number}$'.format(
            **{att: value_fmt[att](orig_data[att].item()) for att in variables + ('type',)}
        )

        ax[0].text(0.5, 0.5, att_str, horizontalalignment='center',
                      verticalalignment='center', transform=ax[0].transAxes,
                      fontsize=mpl.rcParams['axes.titlesize'])
        plt.show()

    from ipywidgets import interactive, IntSlider, FloatSlider, HBox, VBox, Checkbox, Dropdown
    from IPython.display import display

    def plot(image, age, sex, brain_volume, ventricle_volume, lesion_volume, duration, score, slice_number,
             do_age, do_sex, do_brain_volume, do_ventricle_volume, do_lesion_volume, do_duration, do_score, do_slice_number):
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
        if do_score:
            intervention['score'] = score
        if do_slice_number:
            intervention['slice_number'] = slice_number

        plot_intervention(intervention, image)

    w = interactive(plot, image=IntSlider(min=0, max=len(calabresi_test)-1, description='Image #'),
        age=FloatSlider(min=20., max=100., step=1., continuous_update=False, description='Age'),
        do_age=Checkbox(description='do(age)'),
        sex=Dropdown(options=[('female', 0.), ('male', 1.)], description='Sex'),
        do_sex=Checkbox(description='do(sex)'),
        brain_volume=FloatSlider(min=8., max=16., step=1., continuous_update=False, description='Brain Volume (ml):', style={'description_width': 'initial'}),
        do_brain_volume=Checkbox(description='do(brain_volume)'),
        ventricle_volume=FloatSlider(min=1e-5, max=2.5, step=0.5, continuous_update=False, description='Ventricle Volume (ml):', style={'description_width': 'initial'}),
        do_ventricle_volume=Checkbox(description='do(ventricle_volume)'),
        lesion_volume=FloatSlider(min=1e-5, max=2.0, step=0.25, continuous_update=False, description='Lesion Volume (ml):', style={'description_width': 'initial'}),
        do_lesion_volume=Checkbox(description='do(lesion_volume)'),
        duration=FloatSlider(min=1e-5, max=24., step=1., continuous_update=False, description='Duration (y):', style={'description_width': 'initial'}),
        do_duration=Checkbox(description='do(duration)'),
        score=FloatSlider(min=1e-5, max=10., step=1., continuous_update=False, description='Score:', style={'description_width': 'initial'}),
        do_score=Checkbox(description='do(score)'),
        slice_number=FloatSlider(min=100., max=200., step=1., continuous_update=False, description='Slice #:', style={'description_width': 'initial'}),
        do_slice_number=Checkbox(description='do(slice_number)'),
        )

    n = len(variables)
    ui = VBox([w.children[0], VBox([HBox([w.children[i], w.children[i+n]]) for i in range(1,n+1)]), w.children[-1]])
    display(ui)
    w.update()

