#!/usr/bin/env python
"""Generate notebook 08: Full simulation comparison — all G/H/I variants vs actual."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []

# ===== TITLE =====
cells.append(
    nbf.v4.new_markdown_cell(
        "# 08 — Simulation Comparison: All Model G, H, I Variants vs Actual\n"
        "\n"
        "Lap-by-lap position comparison across 4 held-out 2024 test races.\n"
        "\n"
        "**15 models total** (5 per type):\n"
        "- **Model G** (Sequence): SeqGRU_Attn, SeqGRU_Bidir, SeqGRU_Deep, SeqLSTM_Bidir, SeqLSTM_Deep\n"
        "- **Model H** (Delta): LightGBM_Delta, LightGBM_GOSS_Delta, LightGBM_Shallow_Delta, XGBoost_Delta, XGBoost_DART_Delta\n"
        "- **Model I** (Quantile): LightGBM_Quantile, XGBoost_Quantile, GRU_MultiQuantile, FTTransformer_Quantile, MDN_GRU\n"
        "\n"
        "For each driver per race: 3 plots (G, H, I). Each plot shows all 5 variants + actual + MC band (best variant) + pit stops."
    )
)

# ===== IMPORTS =====
cells.append(
    nbf.v4.new_code_cell(
        "import pickle\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from matplotlib.lines import Line2D\n"
        "from matplotlib.patches import Patch\n"
        "from scipy.stats import spearmanr\n"
        "\n"
        "from f1_predictor.data.storage import download_blob\n"
        "from f1_predictor.simulation.defaults import build_circuit_defaults, build_field_median_curves\n"
        "from f1_predictor.simulation.delta_simulator import DeltaRaceSimulator, NoisyModelWrapper\n"
        "from f1_predictor.simulation.engine import RaceSimulator\n"
        "from f1_predictor.simulation.evaluation import (\n"
        "    evaluate_monte_carlo_calibration,\n"
        "    evaluate_simulation,\n"
        ")\n"
        "from f1_predictor.simulation.sequence_simulator import SequenceRaceSimulator\n"
        "\n"
        "import io\n"
        "import torch\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "pd.set_option('display.max_columns', 20)\n"
        "pd.set_option('display.width', 140)\n"
        "%matplotlib inline\n"
        "\n"
        "\n"
        "class CPUUnpickler(pickle.Unpickler):\n"
        "    def find_class(self, module, name):\n"
        "        if module == 'torch.storage' and name == '_load_from_bytes':\n"
        "            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)\n"
        "        return super().find_class(module, name)"
    )
)

# ===== LOAD MODELS =====
cells.append(nbf.v4.new_markdown_cell("## 1. Download and Load All 15 Models"))

cells.append(
    nbf.v4.new_code_cell(
        "MODEL_DIR = Path('data/raw/model')\n"
        "MODEL_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "ALL_MODELS = {\n"
        "    'G': [\n"
        "        'Model_G_SeqGRU_Attn.pkl',\n"
        "        'Model_G_SeqGRU_Bidir.pkl',\n"
        "        'Model_G_SeqGRU_Deep.pkl',\n"
        "        'Model_G_SeqLSTM_Bidir.pkl',\n"
        "        'Model_G_SeqLSTM_Deep.pkl',\n"
        "    ],\n"
        "    'H': [\n"
        "        'Model_H_LightGBM_Delta.pkl',\n"
        "        'Model_H_LightGBM_GOSS_Delta.pkl',\n"
        "        'Model_H_LightGBM_Shallow_Delta.pkl',\n"
        "        'Model_H_XGBoost_Delta.pkl',\n"
        "        'Model_H_XGBoost_DART_Delta.pkl',\n"
        "    ],\n"
        "    'I': [\n"
        "        'Model_I_LightGBM_Quantile.pkl',\n"
        "        'Model_I_XGBoost_Quantile.pkl',\n"
        "        'Model_I_GRU_MultiQuantile.pkl',\n"
        "        'Model_I_FTTransformer_Quantile.pkl',\n"
        "        'Model_I_MDN_GRU.pkl',\n"
        "    ],\n"
        "}\n"
        "\n"
        "BEST = {'G': 'Model_G_SeqLSTM_Bidir.pkl', 'H': 'Model_H_LightGBM_GOSS_Delta.pkl', 'I': 'Model_I_LightGBM_Quantile.pkl'}\n"
        "\n"
        "# Download\n"
        "for mtype, fnames in ALL_MODELS.items():\n"
        "    for fname in fnames:\n"
        "        local = MODEL_DIR / fname\n"
        "        if not local.exists():\n"
        "            print(f'Downloading {fname}...')\n"
        "            download_blob(f'data/raw/model/{fname}', local)\n"
        "        else:\n"
        "            print(f'Already local: {fname}')\n"
        "\n"
        "# Load (CPUUnpickler remaps CUDA tensors to CPU)\n"
        "models = {}\n"
        "for mtype, fnames in ALL_MODELS.items():\n"
        "    for fname in fnames:\n"
        "        with open(MODEL_DIR / fname, 'rb') as f:\n"
        "            m = CPUUnpickler(f).load()\n"
        "        if hasattr(m, 'model_') and hasattr(m.model_, 'cpu'):\n"
        "            m.model_.cpu()\n"
        "            m.model_.eval()\n"
        "        models[fname] = m\n"
        "        short = fname.replace('Model_', '').replace('.pkl', '')\n"
        "        print(f'  Loaded {short}')\n"
        "\n"
        "print(f'\\nAll {len(models)} models loaded.')"
    )
)

# ===== LOAD DATA =====
cells.append(nbf.v4.new_markdown_cell("## 2. Load Data and Build Defaults"))

cells.append(
    nbf.v4.new_code_cell(
        "laps = pd.read_parquet('data/raw/laps/all_laps.parquet')\n"
        "races = pd.read_parquet('data/raw/race/all_races.parquet')\n"
        "\n"
        "circuit_defaults = build_circuit_defaults(laps)\n"
        "field_medians = build_field_median_curves(laps, races)\n"
        "\n"
        "TEST_EVENTS = [\n"
        "    'Bahrain Grand Prix',\n"
        "    'Emilia Romagna Grand Prix',\n"
        "    'Hungarian Grand Prix',\n"
        "    'Mexico City Grand Prix',\n"
        "]\n"
        "\n"
        "DRIVERS_2024 = sorted(races[races['season'] == 2024]['driver_abbrev'].unique())\n"
        "print(f'{len(laps):,} laps | {len(circuit_defaults)} circuits | {len(DRIVERS_2024)} drivers')"
    )
)

# ===== PREPARE RACE DATA =====
cells.append(nbf.v4.new_markdown_cell("## 3. Prepare Test Race Data"))

cells.append(
    nbf.v4.new_code_cell(
        "def build_race_data(event_name, season=2024):\n"
        "    race = races[(races['season'] == season) & (races['event_name'] == event_name)]\n"
        "    race_laps = laps[(laps['season'] == season) & (laps['event_name'] == event_name)].copy()\n"
        "    lap1 = race_laps[race_laps['lap_number'] == 1].set_index('driver_abbrev')\n"
        "\n"
        "    drivers_input, actual_finish = [], {}\n"
        "    for _, row in race.iterrows():\n"
        "        drv = row['driver_abbrev']\n"
        "        tyre = lap1.loc[drv, 'tire_compound'] if drv in lap1.index else 'MEDIUM'\n"
        "        q_valid = [v for v in [row.get('q1_time_sec'), row.get('q2_time_sec'), row.get('q3_time_sec')]\n"
        "                   if v is not None and not np.isnan(v)]\n"
        "        if not q_valid:\n"
        "            continue\n"
        "        drivers_input.append({\n"
        "            'driver': drv, 'grid_position': int(row['grid_position']),\n"
        "            'q1': row.get('q1_time_sec'), 'q2': row.get('q2_time_sec'),\n"
        "            'q3': row.get('q3_time_sec'), 'initial_tyre': tyre,\n"
        "        })\n"
        "        actual_finish[drv] = int(row['finish_position'])\n"
        "\n"
        "    actual_laps = race_laps[['driver_abbrev', 'lap_number', 'position']].dropna(subset=['position']).copy()\n"
        "    actual_laps['position'] = actual_laps['position'].astype(int)\n"
        "    actual_pits = race_laps[race_laps['is_pit_in_lap'] == True].groupby('driver_abbrev')['lap_number'].apply(list).to_dict()\n"
        "    return drivers_input, actual_finish, actual_laps, actual_pits\n"
        "\n"
        "race_data = {}\n"
        "for ev in TEST_EVENTS:\n"
        "    race_data[ev] = build_race_data(ev)\n"
        "    d, f, al, p = race_data[ev]\n"
        "    print(f'{ev}: {len(d)} drivers, {int(al[\"lap_number\"].max())} laps')"
    )
)

# ===== RUN SIMULATIONS =====
cells.append(nbf.v4.new_markdown_cell("## 4. Run All Single-Run Simulations (15 models x 4 races)"))

cells.append(
    nbf.v4.new_code_cell(
        "def make_simulator(fname):\n"
        "    m = models[fname]\n"
        "    if fname.startswith('Model_G'):\n"
        "        ws = getattr(m, 'window_size', 5)\n"
        "        return SequenceRaceSimulator(m, circuit_defaults, window_size=ws)\n"
        "    elif fname.startswith('Model_H'):\n"
        "        return DeltaRaceSimulator(m, circuit_defaults, field_medians)\n"
        "    else:  # Model I — predict() returns q50\n"
        "        return RaceSimulator(m, circuit_defaults)\n"
        "\n"
        "\n"
        "# {event: {filename: SimulationResult}}\n"
        "sim_results = {}\n"
        "# {event: {filename: DataFrame}}  (lap-by-lap)\n"
        "sim_laps = {}\n"
        "\n"
        "for ev in TEST_EVENTS:\n"
        "    sim_results[ev] = {}\n"
        "    sim_laps[ev] = {}\n"
        "    drivers_input = race_data[ev][0]\n"
        "    for mtype, fnames in ALL_MODELS.items():\n"
        "        for fname in fnames:\n"
        "            short = fname.replace('Model_', '').replace('.pkl', '')\n"
        "            try:\n"
        "                sim = make_simulator(fname)\n"
        "                result = sim.simulate(ev, drivers_input)\n"
        "                sim_results[ev][fname] = result\n"
        "                sim_laps[ev][fname] = result.to_dataframe()\n"
        "                print(f'  {ev[:20]:20s} | {short:30s} | OK')\n"
        "            except Exception as e:\n"
        "                print(f'  {ev[:20]:20s} | {short:30s} | ERROR: {e}')\n"
        "\n"
        "print(f'\\nTotal: {sum(len(v) for v in sim_results.values())} successful simulations')"
    )
)

# ===== MC SIMULATIONS =====
cells.append(
    nbf.v4.new_markdown_cell(
        "## 5. Monte Carlo Simulations (Best Variant per Type)\n"
        "\n"
        "30 MC iterations for best G, H, I — capturing full lap-by-lap records for uncertainty bands."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "N_MC = 30\n"
        "NOISE_STD = 0.01\n"
        "SEED = 42\n"
        "\n"
        "\n"
        "class QuantileSamplerWrapper:\n"
        "    def __init__(self, model, rng):\n"
        "        self.model = model\n"
        "        self.rng = rng\n"
        "    def predict(self, X):\n"
        "        q = self.model.predict_quantiles(X)\n"
        "        u = self.rng.uniform(0, 1, size=q.shape[0])\n"
        "        q_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])\n"
        "        return np.array([np.interp(u[i], q_levels, q[i]) for i in range(len(u))])\n"
        "\n"
        "\n"
        "def run_mc(event, drivers_input, model_type, n_mc=N_MC):\n"
        "    rng = np.random.RandomState(SEED)\n"
        "    best_fname = BEST[model_type]\n"
        "    best_model = models[best_fname]\n"
        "    all_dfs = []\n"
        "    for _ in range(n_mc):\n"
        "        if model_type == 'I':\n"
        "            wrapper = QuantileSamplerWrapper(best_model, rng)\n"
        "            sim = RaceSimulator(wrapper, circuit_defaults)\n"
        "        elif model_type == 'H':\n"
        "            wrapper = NoisyModelWrapper(best_model, NOISE_STD, rng)\n"
        "            sim = DeltaRaceSimulator(wrapper, circuit_defaults, field_medians)\n"
        "        else:  # G\n"
        "            wrapper = NoisyModelWrapper(best_model, NOISE_STD, rng)\n"
        "            ws = getattr(best_model, 'window_size', 5)\n"
        "            sim = SequenceRaceSimulator(wrapper, circuit_defaults, window_size=ws)\n"
        "        result = sim.simulate(event, drivers_input)\n"
        "        df = result.to_dataframe()\n"
        "        df['run'] = _\n"
        "        all_dfs.append(df)\n"
        "    return pd.concat(all_dfs, ignore_index=True)\n"
        "\n"
        "\n"
        "# {event: {model_type: DataFrame}}\n"
        "mc_laps = {}\n"
        "for ev in TEST_EVENTS:\n"
        "    mc_laps[ev] = {}\n"
        "    drivers_input = race_data[ev][0]\n"
        "    for mt in ['G', 'H', 'I']:\n"
        "        print(f'  {ev[:20]:20s} | Model {mt} MC({N_MC})...', end=' ', flush=True)\n"
        "        try:\n"
        "            mc_df = run_mc(ev, drivers_input, mt)\n"
        "            mc_laps[ev][mt] = mc_df\n"
        "            print(f'{len(mc_df):,} records')\n"
        "        except Exception as e:\n"
        "            print(f'ERROR: {e}')\n"
        "\n"
        "print('\\nMC simulations complete.')"
    )
)

# ===== METRICS =====
cells.append(nbf.v4.new_markdown_cell("## 6. Metrics"))

cells.append(
    nbf.v4.new_code_cell(
        "def build_eval_df(fname):\n"
        "    rows = []\n"
        "    for ev in TEST_EVENTS:\n"
        "        if fname not in sim_results.get(ev, {}):\n"
        "            continue\n"
        "        result = sim_results[ev][fname]\n"
        "        actual_finish = race_data[ev][1]\n"
        "        for fr in result.final_results:\n"
        "            if fr['driver'] in actual_finish:\n"
        "                rows.append({'event': ev, 'driver': fr['driver'],\n"
        "                             'predicted_pos': fr['position'], 'actual_pos': actual_finish[fr['driver']]})\n"
        "    return pd.DataFrame(rows)\n"
        "\n"
        "\n"
        "print('SINGLE-RUN METRICS — ALL 15 MODELS')\n"
        "print('=' * 100)\n"
        "all_metrics = []\n"
        "for mtype in ['G', 'H', 'I']:\n"
        "    for fname in ALL_MODELS[mtype]:\n"
        "        edf = build_eval_df(fname)\n"
        "        if len(edf) == 0:\n"
        "            continue\n"
        "        m = evaluate_simulation(edf)\n"
        "        short = fname.replace('Model_', '').replace('.pkl', '')\n"
        "        is_best = '***' if fname == BEST[mtype] else ''\n"
        "        m['model'] = f'{short} {is_best}'.strip()\n"
        "        m['type'] = mtype\n"
        "        all_metrics.append(m)\n"
        "\n"
        "mdf = pd.DataFrame(all_metrics)\n"
        "cols = ['type', 'model', 'position_rmse', 'spearman_mean', 'within_3', 'within_5', 'n_races']\n"
        "print(mdf[cols].to_string(index=False, float_format='{:.3f}'.format))\n"
        "\n"
        "# MC metrics\n"
        "print('\\n\\nMONTE CARLO METRICS — BEST VARIANTS')\n"
        "print('=' * 100)\n"
        "for mt in ['G', 'H', 'I']:\n"
        "    rows, details = [], []\n"
        "    for ev in TEST_EVENTS:\n"
        "        if mt not in mc_laps.get(ev, {}):\n"
        "            continue\n"
        "        mc_df = mc_laps[ev][mt]\n"
        "        actual_finish = race_data[ev][1]\n"
        "        max_lap = mc_df['lap_number'].max()\n"
        "        final = mc_df[mc_df['lap_number'] == max_lap]\n"
        "        for drv, grp in final.groupby('driver'):\n"
        "            if drv not in actual_finish:\n"
        "                continue\n"
        "            pos = grp['position'].values\n"
        "            base = {'event': ev, 'driver': drv,\n"
        "                    'predicted_pos': int(np.median(pos)), 'actual_pos': actual_finish[drv]}\n"
        "            rows.append(base.copy())\n"
        "            base.update({'position_p10': int(np.percentile(pos, 10)),\n"
        "                         'position_p25': int(np.percentile(pos, 25)),\n"
        "                         'position_p75': int(np.percentile(pos, 75)),\n"
        "                         'position_p90': int(np.percentile(pos, 90))})\n"
        "            details.append(base)\n"
        "    if rows:\n"
        "        m = evaluate_simulation(pd.DataFrame(rows))\n"
        "        m.update(evaluate_monte_carlo_calibration(details))\n"
        "        print(f'\\nModel {mt} MC({N_MC}) — {BEST[mt].replace(\"Model_\", \"\").replace(\".pkl\", \"\")}:')\n"
        "        for k, v in m.items():\n"
        "            print(f'  {k:20s}: {v:.3f}' if isinstance(v, float) else f'  {k:20s}: {v}')"
    )
)

# ===== PER-RACE METRICS =====
cells.append(
    nbf.v4.new_code_cell(
        "print('PER-RACE SPEARMAN (Best variant per type)')\n"
        "print('=' * 90)\n"
        "pr = []\n"
        "for ev in TEST_EVENTS:\n"
        "    af = race_data[ev][1]\n"
        "    for mt in ['G', 'H', 'I']:\n"
        "        fname = BEST[mt]\n"
        "        if fname not in sim_results.get(ev, {}):\n"
        "            continue\n"
        "        pred = {r['driver']: r['position'] for r in sim_results[ev][fname].final_results}\n"
        "        common = sorted(set(pred) & set(af))\n"
        "        if len(common) < 3:\n"
        "            continue\n"
        "        a, p = [af[d] for d in common], [pred[d] for d in common]\n"
        "        rho, _ = spearmanr(a, p)\n"
        "        rmse = np.sqrt(np.mean((np.array(a) - np.array(p)) ** 2))\n"
        "        pr.append({'event': ev, 'model': f'Model {mt}', 'spearman': rho, 'rmse': rmse})\n"
        "\n"
        "prdf = pd.DataFrame(pr)\n"
        "print(prdf.pivot(index='event', columns='model', values='spearman').to_string(float_format='{:.3f}'.format))\n"
        "print('\\nRMSE:')\n"
        "print(prdf.pivot(index='event', columns='model', values='rmse').to_string(float_format='{:.2f}'.format))"
    )
)

# ===== PLOT HELPERS =====
cells.append(nbf.v4.new_markdown_cell("## 7. Lap-by-Lap Position Plots"))

cells.append(
    nbf.v4.new_code_cell(
        "VARIANT_COLORS = [\n"
        "    '#E53935',  # red (best highlighted)\n"
        "    '#1E88E5',  # blue\n"
        "    '#43A047',  # green\n"
        "    '#FB8C00',  # orange\n"
        "    '#8E24AA',  # purple\n"
        "]\n"
        "MC_COLOR = '#999999'\n"
        "\n"
        "\n"
        "def mc_bands(mc_df, driver):\n"
        "    d = mc_df[mc_df['driver'] == driver]\n"
        "    if len(d) == 0:\n"
        "        return pd.DataFrame()\n"
        "    return d.groupby('lap_number')['position'].agg(\n"
        "        p10=lambda x: np.percentile(x, 10),\n"
        "        p90=lambda x: np.percentile(x, 90),\n"
        "    ).reset_index()\n"
        "\n"
        "\n"
        "def sim_pit_laps(sim_df, driver):\n"
        "    dd = sim_df[sim_df['driver'] == driver].sort_values('lap_number')\n"
        "    return dd[dd['stint'].diff() > 0]['lap_number'].tolist()\n"
        "\n"
        "\n"
        "def plot_driver_model(ax, event, driver, model_type, grid_pos):\n"
        "    \"\"\"Plot one model type (all 5 variants + MC band) for one driver on given axes.\"\"\"\n"
        "    _, _, actual_laps, actual_pits = race_data[event]\n"
        "    da = actual_laps[actual_laps['driver_abbrev'] == driver].sort_values('lap_number')\n"
        "    n_drivers = len(race_data[event][0])\n"
        "    total_laps = circuit_defaults.get(event, {}).get('total_laps', 57)\n"
        "    max_lap = max(total_laps, int(da['lap_number'].max()) if len(da) > 0 else total_laps)\n"
        "\n"
        "    # MC band (best variant)\n"
        "    if model_type in mc_laps.get(event, {}):\n"
        "        b = mc_bands(mc_laps[event][model_type], driver)\n"
        "        if len(b) > 0:\n"
        "            ax.fill_between(b['lap_number'], b['p10'], b['p90'],\n"
        "                            color=MC_COLOR, alpha=0.15, zorder=1, label='MC p10-p90')\n"
        "\n"
        "    # Actual position\n"
        "    if len(da) > 0:\n"
        "        ax.plot(da['lap_number'], da['position'], color='black', lw=2.5, alpha=0.9, zorder=5, label='Actual')\n"
        "\n"
        "    # 5 variant lines (best first so it's most visible)\n"
        "    fnames = ALL_MODELS[model_type]\n"
        "    best_fname = BEST[model_type]\n"
        "    ordered = [best_fname] + [f for f in fnames if f != best_fname]\n"
        "    for ci, fname in enumerate(ordered):\n"
        "        if fname not in sim_laps.get(event, {}):\n"
        "            continue\n"
        "        sdf = sim_laps[event][fname]\n"
        "        ds = sdf[sdf['driver'] == driver].sort_values('lap_number')\n"
        "        if len(ds) == 0:\n"
        "            continue\n"
        "        short = fname.replace('Model_' + model_type + '_', '').replace('.pkl', '')\n"
        "        lw = 2.0 if fname == best_fname else 1.0\n"
        "        alpha = 0.9 if fname == best_fname else 0.6\n"
        "        ax.plot(ds['lap_number'], ds['position'], color=VARIANT_COLORS[ci],\n"
        "                lw=lw, alpha=alpha, ls='--', zorder=3 + (1 if fname == best_fname else 0),\n"
        "                label=short + (' *' if fname == best_fname else ''))\n"
        "\n"
        "    # Actual pit stops\n"
        "    for pl in actual_pits.get(driver, []):\n"
        "        ax.axvline(x=pl, color='black', ls=':', lw=0.8, alpha=0.5, zorder=2)\n"
        "\n"
        "    # Sim pit stops (best variant)\n"
        "    if best_fname in sim_laps.get(event, {}):\n"
        "        for pl in sim_pit_laps(sim_laps[event][best_fname], driver):\n"
        "            ax.axvline(x=pl, color='grey', ls='--', lw=0.5, alpha=0.4, zorder=2)\n"
        "\n"
        "    ax.set_ylim(n_drivers + 0.5, 0.5)\n"
        "    ax.set_xlim(0, max_lap + 1)\n"
        "    ax.set_yticks(range(1, n_drivers + 1, 2))\n"
        "    ax.grid(True, alpha=0.15)\n"
        "    ax.set_title(f'Model {model_type}', fontsize=10, fontweight='bold')\n"
        "    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)\n"
        "\n"
        "\n"
        "def plot_driver(event, driver, grid_pos):\n"
        "    \"\"\"Create a 1x3 figure with G, H, I plots for one driver.\"\"\"\n"
        "    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)\n"
        "    for i, mt in enumerate(['G', 'H', 'I']):\n"
        "        plot_driver_model(axes[i], event, driver, mt, grid_pos)\n"
        "        if i == 0:\n"
        "            axes[i].set_ylabel('Position', fontsize=10)\n"
        "        axes[i].set_xlabel('Lap', fontsize=9)\n"
        "\n"
        "    fig.suptitle(f'{driver} (P{grid_pos}) — {event}', fontsize=13, fontweight='bold')\n"
        "    fig.tight_layout()\n"
        "    plt.show()\n"
        "\n"
        "\n"
        "print('Plotting functions defined.')"
    )
)

# ===== GENERATE PLOT CELLS PER RACE, PER DRIVER =====
# For each race: markdown header + one cell per driver
for event in [
    "Bahrain Grand Prix",
    "Emilia Romagna Grand Prix",
    "Hungarian Grand Prix",
    "Mexico City Grand Prix",
]:
    cells.append(nbf.v4.new_markdown_cell(f"### {event}"))

    # Drivers sorted by grid position — we'll sort dynamically in the cell
    cells.append(
        nbf.v4.new_code_cell(
            f"_ev = '{event}'\n"
            "_drivers_sorted = sorted(race_data[_ev][0], key=lambda d: d['grid_position'])\n"
            "for _d in _drivers_sorted:\n"
            "    plot_driver(_ev, _d['driver'], _d['grid_position'])"
        )
    )

# ===== SUMMARY =====
cells.append(
    nbf.v4.new_markdown_cell(
        "## 8. Summary\n"
        "\n"
        "| Model | Best Variant | RMSE | Spearman | Verdict |\n"
        "|-------|-------------|------|----------|--------|\n"
        "| **H** | LightGBM_GOSS_Delta | ~3.48 | ~0.82 | **Recommended default** |\n"
        "| **G** | SeqLSTM_Bidir | ~3.66 | ~0.80 | Close second |\n"
        "| **I** | LightGBM_Quantile | ~6.99 | ~0.26 | Not recommended for simulation |\n"
        "\n"
        "Model I collapses in autoregressive simulation — quantile sampling noise compounds over 50-70 laps."
    )
)

nb.cells = cells

out_path = Path("notebooks/08_simulation_comparison.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Generated {out_path} with {len(cells)} cells")
