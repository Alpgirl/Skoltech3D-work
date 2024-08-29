from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import matplotlib.colors
import matplotlib.transforms
import numpy as np
import matplotlib.pyplot as plt
import yaml

from skrgbd.evaluation.pathfinder import eval_pathfinder
from skrgbd.utils.logging import logger
from skrgbd.data.dataset.pathfinder import pathfinder


def main():
    description = r"""Draws figures with distnace distributions and saves visualizations to disk."""

    parser = ArgumentParser(description=description)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--scene-name', type=str)
    args = parser.parse_args()

    f'Eval 04 {args.scene_name}' >> logger.debug
    pathfinder.set_dirs(data_root=args.dataset_dir)
    eval_pathfinder.set_dirs(args.results_dir)

    'Draw' >> logger.debug
    fig = draw_figure(args.scene_name)

    config_version = 'v1'
    fig_pdf = eval_pathfinder.evaluation.figures(config_version, args.scene_name).distances
    f'Save to {fig_pdf}' >> logger.debug
    Path(fig_pdf).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight', pad_inches=0)
    # fig.savefig(fig_pdf[:-3] + 'jpg', dpi=100, bbox_inches='tight', pad_inches=0)


def draw_figure(scene_name, light_setup='ambient@best', view_i=53):
    'Setup experiments and labels' >> logger.debug
    method_labels = {'tsdf_fusion': 'TSDF Fusion',
                     'surfel_meshing': 'SurfelMeshing',
                     'routed_fusion': 'Routed Fusion',
                     'colmap': 'COLMAP',
                     'acmp': 'ACMP',
                     'vismvsnet': 'VisMVSNet',
                     'unimvsnet': 'UniMVSNet',
                     'neus': 'NeuS',
                     'azinovic22neural': 'Neural RGBD'}

    experiment_setups = {
        'tsdf_fusion': dict(version='v0.0.0_v3', camera='kinect_v2', light=None),
        'surfel_meshing': dict(version='v1.0.0_v3', camera='kinect_v2', light=None),
        'routed_fusion': dict(version='v0.0.0_v3', camera='kinect_v2', light=None),
        'colmap': dict(version='v1.0.0_v1', camera='tis_right', light='ambient@best'),
        'acmp': dict(version='v1.0.0_v1', camera='tis_right', light='ambient@best'),
        'vismvsnet': dict(version='v1.0.0_trained_on_blendedmvg_v1', camera='tis_right', light='ambient@best'),
        'unimvsnet': dict(version='v1.0.0_authors_checkpoint_v1', camera='tis_right', light='ambient@best'),
        'neus': dict(version='v1.1.0_v1', camera='tis_right', light='ambient@best'),
        'azinovic22neural': dict(version='v0.0.0_v3', camera='kinect_v2@tis_right', light='ambient@best'),
    }
    experiments = dict()
    for method, params in experiment_setups.items():
        results = eval_pathfinder.evaluation[method](params['version'], scene_name, params['camera'], params['light'])
        if Path(results.visualizations.completeness).exists():
            experiments[method] = results
    del experiment_setups

    measure_labels = {'accuracy': 'Accuracy on reconstruction',
                      'surf_accuracy': 'Accuracy on reference',
                      'completeness': 'Completeness on reference'}
    measures = ['accuracy', 'surf_accuracy', 'completeness']

    'Setup style' >> logger.debug
    plt.rc('font', size=80, family='Times New Roman')
    plt.rcParams['axes.spines.left'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.bottom'] = False

    u_title_kwargs = dict(fontsize=8, y=1, verticalalignment='bottom', pad=2)
    b_title_kwargs = dict(fontsize=8, y=0, verticalalignment='top', pad=-2)
    l_label_kwargs = dict(fontsize=8, labelpad=2, verticalalignment='bottom')

    'Setup figure' >> logger.debug
    fig_w = 6.90078125  # CVPR \textwidth in inches
    w_to_h = 1.618
    grid_w, grid_h = 4, len(experiments) + 1
    aspect_k = .98
    fig, axes = plt.subplots(grid_h, grid_w, figsize=(fig_w, fig_w / (grid_w * w_to_h * aspect_k) * grid_h))
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[-1, -2:]:
        fig.delaxes(ax)

    axes[0, 0].set_title('Reconstruction', **u_title_kwargs)
    for measure, ax in zip(measures, axes[0, 1:]):
        ax.set_title(measure_labels[measure], **u_title_kwargs)

    'Load meta' >> logger.debug
    for method, results in experiments.items():
        meta = results.visualizations.meta
        with open(meta, 'r') as file:
            meta = yaml.load(file, yaml.SafeLoader)
        config = meta['config']
        if config['draw_ref']:
            break

    'Put reference' >> logger.debug
    img = results.visualizations.reference
    img = Image.open(img)
    ax = axes[-1, 0]

    w, h = img.size
    extent = ['left', 'right', 'bottom', 'top']
    extent[0] = meta['crop_left_top'][0] - .5
    extent[3] = meta['crop_left_top'][1] - .5
    extent[1] = extent[0] + w
    extent[2] = extent[3] + h

    ax.imshow(img, cmap='gray', vmin=0, vmax=255, extent=extent)
    ax.set_title('Reference scan', **b_title_kwargs)

    'Put distance images' >> logger.debug
    for method_i, (method, results) in enumerate(experiments.items()):
        img = results.visualizations.reconstruction
        img = Image.open(img)
        ax = axes[method_i, 0]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255, extent=extent)
        ax.set_ylabel(f'{method_labels[method]}', **l_label_kwargs)

        for measure_i, measure in enumerate(measures):
            img = results.visualizations[measure]
            img = Image.open(img)
            ax = axes[method_i, measure_i + 1]
            ax.imshow(img, extent=extent)

    'Put RGB' >> logger.debug
    rgb = pathfinder[scene_name].tis_right.rgb.undistorted[(light_setup, view_i)]
    rgb = np.asarray(Image.open(rgb))
    ax = axes[-1, 1]
    ax.imshow(rgb)
    ax.set_title('Photo', **b_title_kwargs)

    'Adjust subplots' >> logger.debug
    bb_wh = np.array([w, h])
    pov = meta['crop_left_top'] + bb_wh / 2
    if (w / h) >= w_to_h:
        bb_wh[1] = w / w_to_h
    else:
        bb_wh[0] = h * w_to_h
    for ax in axes.ravel():
        ax.set_xlim(pov[0] - bb_wh[0] / 2, pov[0] + bb_wh[0] / 2)
        ax.set_ylim(pov[1] + bb_wh[1] / 2, pov[1] - bb_wh[1] / 2)
    plt.subplots_adjust(0, 0, 1, 1, wspace=.01, hspace=0)

    'Put separators' >> logger.debug
    for ax_i in range(1, grid_h):
        line_y = ax_i / grid_h
        line = plt.Line2D([-.02, .99], [line_y, line_y], transform=fig.transFigure, color='black', lw=.3)
        fig.add_artist(line)

    plt.close()
    return fig


def add_cbar(fig, cax, dist_range=(0, 3e-3), color_range=(.02, 1), cmap=plt.cm.hot_r):
    dists = np.linspace(*dist_range, 100)
    dists_n = ((dists - dist_range[0]) * (color_range[1] - color_range[0]) / (dist_range[1] - dist_range[0])
               + color_range[0])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cmap_trunc', cmap(dists_n))

    cbar = fig.colorbar(plt.cm.ScalarMappable(plt.cm.colors.Normalize(*dist_range), cmap=cmap),
                        cax=cax, orientation='horizontal')

    maj_ticks = np.linspace(*dist_range, 4)
    maj_ticks_labels = [f'{round(t * 1000, 3)}' for t in maj_ticks]
    maj_ticks_labels[-1] += ', mm'

    cbar.set_ticks(maj_ticks)
    cbar.ax.set_xticklabels(maj_ticks_labels)
    cbar.ax.tick_params(length=10)

    cbar.minorticks_on()
    cbar.ax.tick_params(length=.618 * 10, which='minor')


if __name__ == '__main__':
    main()
