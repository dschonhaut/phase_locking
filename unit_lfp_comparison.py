"""
unit_lfp_comparison.py

Author
------
Daniel Schonhaut
Computational Memory Lab
University of Pennsylvania
daniel.schonhaut@gmail.com

Description
----------- 
Compare unit spike times to LFP oscillations.

Last Edited
----------- 
5/2/22
"""
import sys
import os.path as op
from glob import glob
# import mkl
# mkl.set_num_threads(1)
import numpy as np
import pandas as pd
from statsmodels.stats.api import multipletests
import astropy.stats.circstats as circstats
sys.path.append('/home1/dscho/code/general')
import data_io as dio
from helper_funcs import Timer, str_replace
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_preproc, events_proc, time_bin_analysis


def unit_to_lfp_phase_locking_gold_view(unit,
                                        freqs=np.arange(1, 31),
                                        n_rois=8,
                                        n_samp_iter=10,
                                        n_perm=1000,
                                        alpha=0.05,
                                        data_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/goldmine/nav',
                                        output_dir=None,
                                        save_output=True,
                                        overwrite=False,
                                        verbose=True):
    """Calculate oscillatory phase-locking gold view events.
    
    Compare golds that were later dug or not dug, respectively.

    Parameters
    ----------
    unit : dict or series
        Contains the spike times vector and identifying info for a unit.
    freqs : array
        Frequencies at which spike-phase relations are analyzed.
    n_rois : int
        Number of meta-regions to assign for categorization.
    n_perm : int
        Number of permutations drawn to construst the null distribution.
        For each permutation, spike times are circ-shifted at random
        within each event, and phase-locking values at each frequency
        are recalculcated across events.
    alpha : float
        Defines the significance threshold for the phase-locking
        empirical p-value.
    data_dir : str
        Filepath to the location where saved inputs are stored.
    output_dir : str | None
        Filepath to the location where the output file is saved.
    save_output : bool
        Output is saved only if True.
    overwrite : bool
        If False and saved output already exists, it is simply returned
        at the top of this function. Otherwise phase-locking is
        calculated and, if save_output is True, any existing output file
        is overwritten.
    verbose : bool
        If True, some info is printed to the standard output.
        
    Returns
    -------
    pl_mrls : dataframe
        Each row corresponds to one unit -> LFPs from one microwire
        bundle.
    """
    timer = Timer()

    # Load the output file if it exists.
    output_b = '{}-{}-{}.pkl'.format(unit['subj_sess'], unit['chan'], unit['unit'])
    if output_dir is None:
        output_dir = op.join(data_dir, 'phase_locking', 'osc2mask')
    output_f = op.join(output_dir, output_b)
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Loading saved output: {}'.format(output_f))
        return dio.open_pickle(output_f)

    # Hard-coded params.
    expmt = 'goldmine'
    game_states = ['Encoding']
    gold_keys = ['dug', 'not_dug']

    # Get a list of LFP ROIs to process.
    mont = spike_preproc.get_montage(unit['subj_sess'])
    roi_map = spike_preproc.roi_mapping(n=n_rois)
    hpc_rois = ['AH', 'MH', 'PH']
    lfp_hemrois = np.unique([unit['hemroi']] + [hemroi for hemroi in mont.keys()
                                                if hemroi[1:] in hpc_rois])
    pl_mrls = []
    for lfp_hemroi in lfp_hemrois:
        # ------------------------------------
        # Data loading

        # Determine which regions we're looking at.
        lfp_roi_gen = roi_map[lfp_hemroi[1:]]
        if unit['roi_gen'] == 'HPC':
            if lfp_roi_gen == 'HPC':
                if unit['hemroi'] == lfp_hemroi:
                    edge = 'hpc-local'
                else:
                    edge = 'hpc-hpc'
            else:
                edge = 'hpc-ctx'
        else:
            if lfp_roi_gen != 'HPC':
                if unit['hemroi'] == lfp_hemroi:
                    edge = 'ctx-local'
                else:
                    edge = 'ctx-ctx'
            else:
                edge = 'ctx-hpc'
        same_hem = unit['hem'] == lfp_hemroi[0]
        same_roi_gen = unit['roi_gen'] == lfp_roi_gen

        # Load event_times.
        event_times = load_event_times(unit['subj_sess'],
                                       expmt=expmt,
                                       game_states=game_states,
                                       data_dir=data_dir)

        # Calculate spike times relative to the start of each event.
        event_time_spikes = load_event_time_spikes(event_times,
                                                   unit['spike_times'])

        # Load phase values for each event.
        basename_lfp = '{}-{}.pkl'.format(unit['subj_sess'], lfp_hemroi)
        phase = dio.open_pickle(op.join(data_dir, 'spectral', 'phase', basename_lfp))
        if (phase.buffer > 0) & (not phase.clip_buffer):
            phase = phase.loc[:, :, :, phase.buffer:phase.time.size-phase.buffer-1]
            phase.attrs['clip_buffer'] = True
        lfp_chans = [chan for chan in phase.chan.values if chan not in [unit['chan']]]
        if unit['chan'] in phase.chan:
            phase = phase.loc[{'chan': lfp_chans}]
        if not np.array_equal(phase.freq.values, freqs):
            phase = phase.loc[{'freq': [freq for freq in phase.freq.values if freq in freqs]}]
        if np.nanmax(phase.values) > np.pi:
            phase -= np.pi  # convert values back to -π to π, with -π being the trough

        # Load the P-episode mask for each event.
        osc_mask = dio.open_pickle(op.join(data_dir, 'p_episode', basename_lfp))
        if (osc_mask.buffer > 0) & (not osc_mask.clip_buffer):
            osc_mask = osc_mask.loc[:, :, :, osc_mask.buffer:osc_mask.time.size-osc_mask.buffer-1]
            osc_mask.attrs['clip_buffer'] = True
        if unit['chan'] in osc_mask.chan:
            osc_mask = osc_mask.loc[{'chan': [chan for chan in osc_mask.chan.values
                                              if chan not in [unit['chan']]]}]
        if not np.array_equal(osc_mask.freq.values, freqs):
            osc_mask = osc_mask.loc[{'freq': [freq for freq in osc_mask.freq.values
                                              if freq in freqs]}]
        osc_mask = osc_mask.sel(gameState='Encoding')  # [trial, chan, freq, time]

        # Load the gold-view mask for each event.        
        basename = '{}-gold_view_mask.pkl'.format(unit['subj_sess'])
        gold_mask = dio.open_pickle(op.join(data_dir, 'events', basename))  # [gold_key][trial, time]

        # Combine gold view and oscillatory masks.
        osc_gold_mask = {key: (gold_mask[key][:, None, None, :] & osc_mask.values)
                         for key in gold_keys}  # [gold_key][trial, chan, freq, time]

        # Calculate P-episode at each frequency, across channels and events.
        peps = {key: 100 * np.nanmean(osc_gold_mask[key], axis=(0, 1, 3))
                for key in gold_keys}
        max_pep = {key: np.nanmax(peps[key]) for key in gold_keys}
        max_pep_freq = {key: np.nanargmax(peps[key]) for key in gold_keys}

        # Get gold view spike phases at each freq. across events,
        # for later dug and not dug golds.
        event_dur = phase.time.size
        spike_phases = {
            key: np.ma.concatenate(
                event_time_spikes.apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase.values[x['event_idx'], :, :, :],
                                               mask=np.invert(osc_gold_mask[key][x['event_idx'], :, :, :]),
                                               event_dur=event_dur,
                                               circshift=False),
                    axis=1).tolist(), axis=-1)
            for key in gold_keys}  # [gold_key][chan, freq, spike]


        # Draw spikes from dug and not_dug gold view times.

        # Calculate mean resultant lengths for each channel and frequency.
        n_freq = phase.freq.size
        n_chan = phase.chan.size
        mrls = {key: np.nanmean([[[circstats.circmoment(np.random.choice(
                                   spike_phases[key][iChan, iFreq, :].compressed(),
                                   size=n_spike_samp, replace=True))[1]
                                   for iFreq in range(n_freq)]
                                  for iChan in range(n_chan)]
                                 for ii in range(n_samp_iter)],
                                axis=(0, 1)).astype(np.float32)  # [freq]
                for key in gold_keys}

        for key in gold_keys:
            

            # ------------------------------------
            # Calculate real phase-locking

            # Get a masked array of spike phases, across events, during active
            # oscillations at each channel and frequency.
            
            spike_phases = np.ma.concatenate(event_time_spikes.apply(
                lambda x: get_spike_phases(x['spike_times'],
                                           phase.values[x['event_idx'], :, :, :],
                                           mask=np.invert(osc_gold_mask[key].values[x['event_idx'], :, :, :]),
                                           event_dur=event_dur,
                                           circshift=False),
                axis=1).tolist(), axis=-1) # chan x freq x spike

            # Calculate mean resultant lengths for each channel and frequency.
            mrls = np.nanmean([[circstats.circmoment(spike_phases[iChan, iFreq, :].compressed())[1]
                                for iFreq in range(spike_phases.shape[1])]
                               for iChan in range(spike_phases.shape[0])], axis=0).astype(np.float32) # (freq,)

            # Log how many spikes were counted per frequency,
            # taking the mean across channels.
            n_spikes_mask = np.nanmean([[spike_phases[iChan, iFreq, :].flatten().compressed().size
                                         for iFreq in range(spike_phases.shape[1])]
                                        for iChan in range(spike_phases.shape[0])], axis=0).astype(np.float32) # (freq,)

            # Get the preferred phase at each frequency, across channels.
            pref_phases = np.array([circstats.circmoment(spike_phases[:, iFreq, :].flatten().compressed())[0]
                                    for iFreq in range(spike_phases.shape[1])]).astype(np.float32) # (freq,)

            # ------------------------------------
            # Calculate null phase-locking
            mrls_null = []
            n_spikes_mask_null = []
            for iPerm in range(n_perm):
                # Get a masked array of spike phases, across events, during active
                # oscillations at each channel and frequency.
                _spike_phases_null = np.ma.concatenate(event_time_spikes.apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase[x['event_idx'], :, :, :],
                                               mask=np.invert(osc_mask[key][x['event_idx'], :, :, :]),
                                               event_dur=event_dur,
                                               circshift=True),
                    axis=1).tolist(), axis=-1) # chan x freq x spike

                # Calculate mean resultant lengths for each channel and frequency.
                _mrls_null = np.nanmean([[circstats.circmoment(_spike_phases_null[iChan, iFreq, :].compressed())[1]
                                          for iFreq in range(_spike_phases_null.shape[1])]
                                         for iChan in range(_spike_phases_null.shape[0])], axis=0).astype(np.float32)
                mrls_null.append(_mrls_null.tolist())

                _n_spikes_mask_null = np.nanmean([[_spike_phases_null[iChan, iFreq, :].flatten().compressed().size
                                                   for iFreq in range(_spike_phases_null.shape[1])]
                                                  for iChan in range(_spike_phases_null.shape[0])], axis=0).astype(np.float32)
                n_spikes_mask_null.append(_n_spikes_mask_null)

            mrls_null = np.array(mrls_null) # (perm, freq)
            n_spikes_mask_null = np.array(n_spikes_mask_null) # (perm, freq)

            # ------------------------------------
            # Statistics

            # Z-score MRLs against the null distribution,
            # and calculate empirical P-values.
            mean_mrls_null = np.nanmean(mrls_null, axis=0)
            std_mrls_null = np.nanstd(mrls_null, axis=0)
            z_mrls = (mrls - mean_mrls_null) / std_mrls_null # (freq,)
            z_mrls_null = (mrls_null - mean_mrls_null[None, :]) / std_mrls_null[None, :] # (perm, freq)
            max_z_mrl = np.nanmax(z_mrls)
            max_z_mrl_freq = freqs[np.nanargmax(z_mrls)]
            pval = (1 + np.nansum(np.nanmax(z_mrls_null, axis=1) >= max_z_mrl)) / (1 + n_perm)
            sig = pval < alpha

            mean_n_spikes_mask_null = np.nanmean(n_spikes_mask_null, axis=0)
            std_n_spikes_mask_null = np.nanstd(n_spikes_mask_null, axis=0)
            z_n_spikes_mask = (n_spikes_mask - mean_n_spikes_mask_null) / std_n_spikes_mask_null

            # ------------------------------------
            # Add results to dataframe.
            pl_mrls.append([unit['subj'],
                            unit['subj_sess'],
                            '{}-{}'.format(unit['chan'], unit['unit']),
                            unit['hemroi'],
                            unit['roi_gen'],
                            unit['n_spikes'],
                            unit['fr'],
                            lfp_hemroi,
                            lfp_roi_gen,
                            lfp_chans,
                            edge,
                            same_hem,
                            same_roi_gen,
                            key,
                            peps,
                            max_pep,
                            max_pep_freq,
                            n_spikes_mask,
                            n_spikes_mask_null,
                            mrls,
                            mrls_null,
                            pref_phases,
                            z_n_spikes_mask,
                            z_mrls,
                            max_z_mrl,
                            max_z_mrl_freq,
                            pval,
                            sig])

    # Create the output dataframe.
    cols = ['subj', 'subj_sess', 'unit', 'unit_hemroi', 'unit_roi_gen', 'n_spikes', 'fr',
            'lfp_hemroi', 'lfp_roi_gen', 'lfp_chans', 'edge', 'same_hem', 'same_roi_gen',
            'mask', 'peps', 'max_pep', 'max_pep_freq', 'n_spikes_mask', 'n_spikes_mask_null',
            'mrls', 'mrls_null', 'pref_phases',
            'z_n_spikes_mask', 'z_mrls', 'max_z_mrl', 'max_z_mrl_freq', 'pval', 'sig']
    pl_mrls = pd.DataFrame(pl_mrls, columns=cols)

    # Save the output.
    if save_output:
        dio.save_pickle(pl_mrls, output_f, verbose)
    
    if verbose:
        print('pl_mrls: {}'.format(pl_mrls.shape))
        print(timer)
        
    return pl_mrls


def unit_to_lfp_phase_locking_goldmine_game_states(
    unit,
    n_rois=8,
    keep_same_hem=[True, False],
    keep_edges=['hpc-local', 'hpc-hpc', 'hpc-ctx', 'ctx-local', 'ctx-hpc', 'ctx-ctx'],
    exclude_gen_rois=[],
    mask_phase=True,
    match_spikes=True,
    n_perm=1000,
    alpha=0.05,
    data_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/goldmine',
    output_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/goldmine/game_states/phase_locking',
    save_output=True,
    overwrite=True,
    verbose=True
):
    """Calculate unit's phase-locking to LFP oscillations
    across all 4 Goldmine game states (Delay1, Encoding, Delay2, Retrieval)
    and in each state, separately (matched for spikes within each
    channel, frequency pair).

    Parameters
    ----------
    unit : dict or series
        Contains the spike times vector and identifying info for a unit.
    expmt : str, 'goldmine' | 'ycab'
        Indicates which dataset we're working with.
    game_states : array
        Event intervals to include in the analysis.
    freqs : array
        Frequencies at which spike-phase relations are analyzed.
    n_rois : int
        Number of meta-regions to assign for categorization.
    keep_same_hem : list[bool]
        Which hemispheric connections to process. Default is to run both
        ipsilateral and contralateral.
    keep_edges : list[str]
        Define the edges to be processed. Default is to analyze all
        electrode pairs.
    exclude_gen_rois : list[str]
        Exclude LFP gen_roi regions in this list.
    match_spikes : bool
        If True, will find the minimum number of spikes across game
        states at each channel and frequency, and subsample that many
        spikes (without replacement) from each mask.
    n_perm : int
        Number of permutations drawn to construst the null distribution.
        For each permutation, spike times are circ-shifted at random
        within each event, and phase-locking values at each frequency
        are recalculcated across events.
    alpha : float
        Defines the significance threshold for the phase-locking
        empirical p-value.
    data_dir : str
        Filepath to the location where saved inputs are stored.
    output_dir : str | None
        Filepath to the location where the output file is saved.
    save_output : bool
        Output is saved only if True.
    overwrite : bool
        If False and saved output already exists, it is simply returned
        at the top of this function. Otherwise phase-locking is
        calculated and, if save_output is True, any existing output file
        is overwritten.
    verbose : bool
        If True, some info is printed to the standard output.
        
    Returns
    -------
    pl_mrls : dataframe
        Each row corresponds to one unit -> LFPs from one microwire
        bundle.
    """
    timer = Timer()
    expmt = 'goldmine'
    game_states = ['Delay1', 'Encoding', 'Delay2', 'Retrieval']

    # Load the output file if it exists.
    output_b = '{}-{}-{}.pkl'.format(unit['subj_sess'], unit['chan'], unit['unit'])
    if output_dir is None:
        output_dir = op.join(data_dir, 'game_states', 'phase_locking')
    output_f = op.join(output_dir, output_b)
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Loading saved output: {}'.format(output_f))
        return dio.open_pickle(output_f)

    # Get a list of LFP ROIs to process.
    mont = spike_preproc.get_montage(unit['subj_sess'])
    roi_map = spike_preproc.roi_mapping(n=n_rois)
    process_pairs = _get_process_pairs(mont,
                                       unit['hemroi'],
                                       keep_same_hem,
                                       keep_edges,
                                       exclude_gen_rois,
                                       roi_map)

    pl_mrls = []
    for (unit_hemroi, lfp_hemroi) in process_pairs:
        # ------------------------------------
        # Data loading

        # Determine which regions we're looking at.
        same_hem = (unit_hemroi[0] == lfp_hemroi[0])
        edge = _get_edge(unit_hemroi, lfp_hemroi)
        unit_roi_gen = roi_map[unit_hemroi[1:]]
        lfp_roi_gen = roi_map[lfp_hemroi[1:]]
        same_roi_gen = (unit_roi_gen == lfp_roi_gen)        

        event_times = {}
        event_time_spikes = {}
        phase = {}
        osc_mask = {}
        for game_state in game_states:
            # Load event_times.
            event_times[game_state] = load_event_times(unit['subj_sess'],
                                                       expmt=expmt,
                                                       game_states=[game_state])

            # Calculate spike times relative to the start of each event.
            event_time_spikes[game_state] = load_event_time_spikes(event_times[game_state],
                                                                   unit['spike_times'])

            delay_and_nav = np.all((np.any(np.isin(game_states, ['Delay1', 'Delay2'])),
                                    np.any(np.isin(game_states, ['Encoding', 'Retrieval']))))

            # Load phase values for each event.
            _data_dir = op.join(data_dir, 'delay' if 'Delay' in game_state else 'nav')
            basename_lfp = '{}-{}.pkl'.format(unit['subj_sess'], lfp_hemroi)
            phase[game_state] = dio.open_pickle(op.join(_data_dir, 'spectral', 'phase', basename_lfp))
            phase[game_state] = phase[game_state].sel(gameState=game_state)
            if (phase[game_state].buffer > 0) & (not phase[game_state].clip_buffer):
                phase[game_state] = phase[game_state].loc[:, :, :, phase[game_state].buffer:phase[game_state].time.size-phase[game_state].buffer-1]
                phase[game_state].attrs['clip_buffer'] = True
            lfp_chans = [chan for chan in phase[game_state].chan.values
                         if chan not in [unit['chan']]]
            if unit['chan'] in phase[game_state].chan:
                phase[game_state] = phase[game_state].loc[{'chan': lfp_chans}]
            freqs = phase[game_state].freq.values
            if not np.array_equal(phase[game_state].freq.values, freqs):
                phase[game_state] = phase[game_state].loc[{'freq': [freq for freq in phase[game_state].freq.values
                                                                    if freq in freqs]}]
            if np.nanmax(phase[game_state].values) > np.pi:
                phase[game_state] -= np.pi # convert values back to -π to π, with -π being the trough.

            # Load the P-episode mask for each event.
            osc_mask[game_state] = dio.open_pickle(op.join(_data_dir, 'p_episode', basename_lfp))
            osc_mask[game_state] = osc_mask[game_state].sel(gameState=game_state)
            if (osc_mask[game_state].buffer > 0) & (not osc_mask[game_state].clip_buffer):
                osc_mask[game_state] = osc_mask[game_state].loc[:, :, :, osc_mask[game_state].buffer:osc_mask[game_state].time.size-osc_mask[game_state].buffer-1]
                osc_mask[game_state].attrs['clip_buffer'] = True
            if unit['chan'] in osc_mask[game_state].chan:
                osc_mask[game_state] = osc_mask[game_state].loc[{'chan': lfp_chans}]
            if not np.array_equal(osc_mask[game_state].freq.values, freqs):
                osc_mask[game_state] = osc_mask[game_state].loc[{'freq': [freq for freq in osc_mask[game_state].freq.values
                                                                          if freq in freqs]}]

            # Ensure that phase, oscillation mask, and spike_times are all in the same event order.
            assert np.array_equal(phase[game_state].trial.values, osc_mask[game_state].trial.values)
            assert np.array_equal(phase[game_state].trial.values, event_time_spikes[game_state]['trial'].values)

        # Remove channels that aren't common to all game states.
        keep_chans = list(set(phase['Delay1'].chan.values) & set(phase['Encoding'].chan.values))
        for game_state in game_states:
            phase[game_state] = phase[game_state].loc[{'chan': keep_chans}]
            osc_mask[game_state] = osc_mask[game_state].loc[{'chan': keep_chans}]
        assert np.array_equal(phase['Delay2'].chan.values, osc_mask['Retrieval'].chan.values)

        # Calculate P-episode at each frequency, across channels and events.
        peps = {}
        max_pep = {}
        max_pep_freq = {}
        for game_state in game_states:
            peps[game_state] = 100 * osc_mask[game_state].mean(dim=('trial', 'chan', 'time')).values
            max_pep[game_state] = peps[game_state].max()
            max_pep_freq[game_state] = freqs[peps[game_state].argmax()]
        for key in ['all', 'all_matched']:
            peps[key] = np.mean([peps['Delay1'],
                                 peps['Encoding'], peps['Encoding'], peps['Encoding'],
                                 peps['Delay2'],
                                 peps['Retrieval'], peps['Retrieval'], peps['Retrieval']],
                                axis=0)
            max_pep[key] = peps[key].max()
            max_pep_freq[key] = peps[key].argmax()

        # ------------------------------------
        # Calculate real phase-locking
        n_chan = phase['Delay1'].chan.size
        n_freq = phase['Delay1'].freq.size
        mask_keys = ['all', 'all_matched'] + game_states

        # Get a masked array of spike phases, across events, during active
        # oscillations at each channel and frequency.
        spike_phases = {}
        for game_state in game_states:
            event_dur = phase[game_state].time.size
            if mask_phase:
                spike_phases[game_state] = np.ma.concatenate(event_time_spikes[game_state].apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase[game_state].values[x['event_idx'], :, :, :],
                                               mask=np.invert(osc_mask[game_state].values[x['event_idx'], :, :, :]),
                                               event_dur=event_dur,
                                               circshift=False),
                    axis=1).tolist(), axis=-1) # chan x freq x spike
            else:
                spike_phases[game_state] = np.ma.concatenate(event_time_spikes[game_state].apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase[game_state].values[x['event_idx'], :, :, :],
                                               mask=None,
                                               event_dur=event_dur,
                                               circshift=False),
                    axis=1).tolist(), axis=-1) # chan x freq x spike

        # Calculate mean resultant lengths for each channel and frequency.
        mrls = {}
        n_spikes_mask = {}
        pref_phases = {}
        mrls['all'] = np.nanmean([[
            circstats.circmoment(np.concatenate([spike_phases[game_state][iChan, iFreq, :].compressed()
                                                 for game_state in game_states]))[1]
            for iFreq in range(n_freq)]
            for iChan in range(n_chan)],
            axis=0).astype(np.float32) # (freq,)

        # Log how many spikes were counted per frequency,
        # taking the mean across channels.
        n_spikes_mask['all'] = np.nanmean([[
            np.concatenate([spike_phases[game_state][iChan, iFreq, :].compressed()
                            for game_state in game_states]).size
            for iFreq in range(n_freq)]
            for iChan in range(n_chan)],
            axis=0).astype(np.float32) # (freq,)

        # Get the preferred phase at each frequency, across channels.
        pref_phases['all'] = np.array([
            circstats.circmoment(np.concatenate([spike_phases[game_state][:, iFreq, :].flatten().compressed()
                                                 for game_state in game_states]))[0]
            for iFreq in range(n_freq)]).astype(np.float32) # (freq,)

        # Match the number of spikes sampled between game states, at each channel and frequency.
        if match_spikes:
            n_spikes = np.min([np.count_nonzero(~spike_phases[game_state].mask, axis=2)
                               for game_state in game_states], axis=0) # (chan, freq)
            for game_state in game_states:
                spike_phases[game_state].mask = subsample_mask(spike_phases[game_state].mask, size=n_spikes)

        # Calculate mean resultant lengths for each channel and frequency.
        mrls['all_matched'] = np.nanmean([[
            circstats.circmoment(np.concatenate([spike_phases[game_state][iChan, iFreq, :].compressed()
                                                 for game_state in game_states]))[1]
            for iFreq in range(n_freq)]
            for iChan in range(n_chan)],
            axis=0).astype(np.float32) # (freq,)

        # Log how many spikes were counted per frequency,
        # taking the mean across channels.
        n_spikes_mask['all_matched'] = np.nanmean([[
            np.concatenate([spike_phases[game_state][iChan, iFreq, :].compressed()
                            for game_state in game_states]).size
            for iFreq in range(n_freq)]
            for iChan in range(n_chan)],
            axis=0).astype(np.float32) # (freq,)

        # Get the preferred phase at each frequency, across channels.
        pref_phases['all_matched'] = np.array([
            circstats.circmoment(np.concatenate([
                spike_phases[game_state][:, iFreq, :].flatten().compressed()
                for game_state in game_states]))[0]
            for iFreq in range(n_freq)]).astype(np.float32) # (freq,)

        for game_state in game_states:
            # Calculate mean resultant lengths for each channel and frequency.
            mrls[game_state] = np.nanmean([[
                circstats.circmoment(spike_phases[game_state][iChan, iFreq, :].compressed())[1]
                for iFreq in range(spike_phases[game_state].shape[1])]
                for iChan in range(spike_phases[game_state].shape[0])],
                axis=0).astype(np.float32) # (freq,)

            # Log how many spikes were counted per frequency,
            # taking the mean across channels.
            n_spikes_mask[game_state] = np.nanmean([[spike_phases[game_state][iChan, iFreq, :].flatten().compressed().size
                                                     for iFreq in range(spike_phases[game_state].shape[1])]
                                                    for iChan in range(spike_phases[game_state].shape[0])], axis=0).astype(np.float32) # (freq,)

            # Get the preferred phase at each frequency, across channels.
            pref_phases[game_state] = np.array([circstats.circmoment(spike_phases[game_state][:, iFreq, :].flatten().compressed())[0]
                                                for iFreq in range(spike_phases[game_state].shape[1])]).astype(np.float32) # (freq,)

        # ------------------------------------
        # Calculate null phase-locking
        mrls_null = {key: [] for key in mask_keys}
        n_spikes_mask_null = {key: [] for key in mask_keys}
        for iPerm in range(n_perm):
            # Get a masked array of spike phases, across events, during active
            # oscillations at each channel and frequency.
            _spike_phases_null = {}
            for game_state in game_states:
                event_dur = phase[game_state].time.size
                if mask_phase:
                    _spike_phases_null[game_state] = np.ma.concatenate(event_time_spikes[game_state].apply(
                        lambda x: get_spike_phases(x['spike_times'],
                                                   phase[game_state].values[x['event_idx'], :, :, :],
                                                   mask=np.invert(osc_mask[game_state].values[x['event_idx'], :, :, :]),
                                                   event_dur=event_dur,
                                                   circshift=True),
                        axis=1).tolist(), axis=-1) # chan x freq x spike
                else:
                    _spike_phases_null[game_state] = np.ma.concatenate(event_time_spikes[game_state].apply(
                        lambda x: get_spike_phases(x['spike_times'],
                                                   phase[game_state].values[x['event_idx'], :, :, :],
                                                   mask=None,
                                                   event_dur=event_dur,
                                                   circshift=True),
                        axis=1).tolist(), axis=-1) # chan x freq x spike

            # Calculate mean resultant lengths for each channel and frequency.
            mrls_null['all'].append(np.nanmean([[
                circstats.circmoment(np.concatenate([_spike_phases_null[game_state][iChan, iFreq, :].compressed()
                                                     for game_state in game_states]))[1]
                for iFreq in range(n_freq)]
                for iChan in range(n_chan)],
                axis=0).astype(np.float32).tolist()) # (freq,)

            # Log how many spikes were counted per frequency,
            # taking the mean across channels.
            n_spikes_mask_null['all'].append(np.nanmean([[
                np.concatenate([_spike_phases_null[game_state][iChan, iFreq, :].compressed()
                                for game_state in game_states]).size
                for iFreq in range(n_freq)]
                for iChan in range(n_chan)],
                axis=0).astype(np.float32).tolist()) # (freq,)

            # Match the number of spikes sampled between game states, at each channel and frequency.
            if match_spikes:
                n_spikes_null = np.min([np.count_nonzero(~_spike_phases_null[game_state].mask, axis=2)
                                        for game_state in game_states], axis=0) # (chan, freq)
                for game_state in game_states:
                    _spike_phases_null[game_state].mask = subsample_mask(_spike_phases_null[game_state].mask, size=n_spikes_null)

            # Calculate mean resultant lengths for each channel and frequency.
            mrls_null['all_matched'].append(np.nanmean([[
                circstats.circmoment(np.concatenate([_spike_phases_null[game_state][iChan, iFreq, :].compressed()
                                                     for game_state in game_states]))[1]
                for iFreq in range(n_freq)]
                for iChan in range(n_chan)],
                axis=0).astype(np.float32).tolist()) # (freq,)

            # Log how many spikes were counted per frequency,
            # taking the mean across channels.
            n_spikes_mask_null['all_matched'].append(np.nanmean([[
                np.concatenate([_spike_phases_null[game_state][iChan, iFreq, :].compressed()
                                for game_state in game_states]).size
                for iFreq in range(n_freq)]
                for iChan in range(n_chan)],
                axis=0).astype(np.float32).tolist()) # (freq,)

            for game_state in game_states:
                # Calculate mean resultant lengths for each channel and frequency.
                mrls_null[game_state].append(np.nanmean([[
                    circstats.circmoment(_spike_phases_null[game_state][iChan, iFreq, :].compressed())[1]
                    for iFreq in range(_spike_phases_null[game_state].shape[1])]
                    for iChan in range(_spike_phases_null[game_state].shape[0])],
                    axis=0).astype(np.float32).tolist()) # (freq,)

                # Log how many spikes were counted per frequency,
                # taking the mean across channels.
                n_spikes_mask_null[game_state].append(np.nanmean([[
                    _spike_phases_null[game_state][iChan, iFreq, :].flatten().compressed().size
                    for iFreq in range(_spike_phases_null[game_state].shape[1])]
                    for iChan in range(_spike_phases_null[game_state].shape[0])],
                    axis=0).astype(np.float32).tolist()) # (freq,)

        mrls_null = {key: np.array(mrls_null[key]) for key in mrls_null} # (perm, freq)
        n_spikes_mask_null = {key: np.array(n_spikes_mask_null[key]) for key in n_spikes_mask_null} # (perm, freq)

        # ------------------------------------
        # Statistics

        # Z-score MRLs against the null distribution,
        # and calculate empirical P-values.
        mean_mrls_null = {key: np.nanmean(mrls_null[key], axis=0) for key in mask_keys}
        std_mrls_null = {key: np.nanstd(mrls_null[key], axis=0) for key in mask_keys}
        z_mrls = {key: (mrls[key] - mean_mrls_null[key]) / std_mrls_null[key] # (freq,)
                  for key in mask_keys}
        z_mrls_null = {key: (mrls_null[key] - mean_mrls_null[key][None, :])
                            / std_mrls_null[key][None, :] # (perm, freq)
                       for key in mask_keys}
        max_z_mrl = {key: np.nanmax(z_mrls[key]) for key in mask_keys}
        max_z_mrl_freq = {key: freqs[np.nanargmax(z_mrls[key])] for key in mask_keys}
        pval = {key: (1 + np.nansum(np.nanmax(z_mrls_null[key], axis=1) >= max_z_mrl[key]))
                     / (1 + n_perm)
                for key in mask_keys}
        sig = {key: pval[key] < alpha for key in mask_keys}

        mean_n_spikes_mask_null = {key: np.nanmean(n_spikes_mask_null[key], axis=0)
                                   for key in mask_keys}
        std_n_spikes_mask_null = {key: np.nanstd(n_spikes_mask_null[key], axis=0)
                                  for key in mask_keys}
        z_n_spikes_mask = {key: (n_spikes_mask[key] - mean_n_spikes_mask_null[key])
                                / std_n_spikes_mask_null[key]
                           for key in mask_keys}

        # ------------------------------------
        # Add results to dataframe.
        for key in mask_keys:
            pl_mrls.append([
                unit['subj'],
                unit['subj_sess'],
                '{}-{}'.format(unit['chan'], unit['unit']),
                unit['hemroi'],
                unit['roi_gen'],
                unit['n_spikes'],
                unit['fr'],
                lfp_hemroi,
                lfp_roi_gen,
                lfp_chans,
                edge,
                same_hem,
                same_roi_gen,
                key,
                peps[key],
                max_pep[key],
                max_pep_freq[key],
                n_spikes_mask[key],
                n_spikes_mask_null[key],
                mrls[key],
                mrls_null[key],
                mean_mrls_null[key].tolist(),
                std_mrls_null[key].tolist(),
                pref_phases[key],
                z_n_spikes_mask[key],
                z_mrls[key],
                max_z_mrl[key],
                max_z_mrl_freq[key],
                pval[key],
                sig[key]
            ])

    # Create the output dataframe.
    cols = ['subj', 'subj_sess', 'unit', 'unit_hemroi', 'unit_roi_gen', 'n_spikes', 'fr',
            'lfp_hemroi', 'lfp_roi_gen', 'lfp_chans', 'edge', 'same_hem', 'same_roi_gen',
            'mask', 'peps', 'max_pep', 'max_pep_freq', 'n_spikes_mask', 'n_spikes_mask_null',
            'mrls', 'mrls_null', 'mean_mrls_null', 'std_mrls_null', 'pref_phases',
            'z_n_spikes_mask', 'z_mrls', 'max_z_mrl', 'max_z_mrl_freq', 'pval', 'sig']
    pl_mrls = pd.DataFrame(pl_mrls, columns=cols)

    # Save the output.
    if save_output:
        dio.save_pickle(pl_mrls, output_f, verbose)

    if verbose:
        print('pl_mrls: {}'.format(pl_mrls.shape))
        print(timer)

    return pl_mrls


def unit_to_lfp_phase_locking_osc2mask(unit,
                                       expmt='goldmine',
                                       game_states=['Encoding', 'Retrieval'],
                                       freqs=np.arange(1, 31),
                                       n_rois=8,
                                       keep_same_hem=[True, False],
                                       keep_edges=['hpc-hpc', 'hpc-ctx', 'ctx-hpc', 'ctx-ctx'],
                                       exclude_gen_rois=[],
                                       match_spikes=False,
                                       n_perm=1000,
                                       alpha=0.05,
                                       data_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/goldmine/nav',
                                       output_dir=None,
                                       save_output=True,
                                       overwrite=False,
                                       verbose=True):
    """Calculate unit's phase-locking to distal LFP oscillations when
    local oscillations are present versus absent.

    Parameters
    ----------
    unit : dict or series
        Contains the spike times vector and identifying info for a unit.
    expmt : str, 'goldmine' | 'ycab'
        Indicates which dataset we're working with.
    game_states : array
        Event intervals to include in the analysis.
    freqs : array
        Frequencies at which spike-phase relations are analyzed.
    n_rois : int
        Number of meta-regions to assign for categorization.
    keep_same_hem : list[bool]
        Which hemispheric connections to process. Default is to run both
        ipsilateral and contralateral.
    keep_edges : list[str]
        Define the edges to be processed. Default is to analyze all
        electrode pairs.
    exclude_gen_rois : list[str]
        Exclude LFP gen_roi regions in this list.
    match_spikes : bool
        If True, will find the minimum number of spikes between
        'lfp_and_unit' and 'lfp_not_unit' masks at each channel and
        frequency, and subsample that many spikes (without replacement)
        from each mask.
    n_perm : int
        Number of permutations drawn to construst the null distribution.
        For each permutation, spike times are circ-shifted at random
        within each event, and phase-locking values at each frequency
        are recalculcated across events.
    alpha : float
        Defines the significance threshold for the phase-locking
        empirical p-value.
    data_dir : str
        Filepath to the location where saved inputs are stored.
    output_dir : str | None
        Filepath to the location where the output file is saved.
    save_output : bool
        Output is saved only if True.
    overwrite : bool
        If False and saved output already exists, it is simply returned
        at the top of this function. Otherwise phase-locking is
        calculated and, if save_output is True, any existing output file
        is overwritten.
    verbose : bool
        If True, some info is printed to the standard output.
        
    Returns
    -------
    pl_mrls : dataframe
        Each row corresponds to one unit -> LFPs from one microwire
        bundle.
    """
    timer = Timer()

    # Load the output file if it exists.
    output_b = '{}-{}-{}.pkl'.format(unit['subj_sess'], unit['chan'], unit['unit'])
    if output_dir is None:
        output_dir = op.join(data_dir, 'phase_locking', 'osc2mask_{}perm'.format(n_perm))
    output_f = op.join(output_dir, output_b)
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Loading saved output: {}'.format(output_f))
        return dio.open_pickle(output_f)

    # Get a list of LFP ROIs to process.
    if expmt == 'goldmine':
        mont = spike_preproc.get_montage(unit['subj_sess'])
    elif expmt == 'ycab':
        subj_df = _get_subj_df()
        subj_df['subj_sess'] = subj_df['subj_sess'].apply(
            lambda x: str_replace(x, {'env': 'ses', '1a': '1'}))
        mont = (subj_df.query("(subj_sess=='{}')".format(unit['subj_sess']))
                       .groupby(['location'])['chan']
                       .apply(lambda x: np.array([int(chan) for chan in x])))
    roi_map = spike_preproc.roi_mapping(n=n_rois)
    process_pairs = _get_process_pairs(mont,
                                       unit['hemroi'],
                                       keep_same_hem,
                                       keep_edges,
                                       exclude_gen_rois,
                                       roi_map)

    pl_mrls = []
    for (unit_hemroi, lfp_hemroi) in process_pairs:
        # ------------------------------------
        # Data loading

        # Determine which regions we're looking at.
        comp_hemroi = None
        if unit_hemroi == lfp_hemroi:
            for (_unit_hemroi, _lfp_hemroi) in process_pairs:
                if roi_map[_lfp_hemroi[1:]] == 'HPC':
                    comp_hemroi = _lfp_hemroi
                    break
            if comp_hemroi is None:
                continue
        else:
            comp_hemroi = unit['hemroi']
        comp_roi_gen = roi_map[comp_hemroi[1:]]
        same_hem = (unit_hemroi[0] == lfp_hemroi[0])
        edge = _get_edge(unit_hemroi, lfp_hemroi)
        unit_roi_gen = roi_map[unit_hemroi[1:]]
        lfp_roi_gen = roi_map[lfp_hemroi[1:]]
        same_roi_gen = (unit_roi_gen == lfp_roi_gen)        

        # Load event_times.
        event_times = load_event_times(unit['subj_sess'],
                                       expmt=expmt,
                                       game_states=game_states,
                                       data_dir=data_dir)

        # Calculate spike times relative to the start of each event.
        event_time_spikes = load_event_time_spikes(event_times,
                                                   unit['spike_times'])

        # Load phase values for each event.
        basename_lfp = '{}-{}.pkl'.format(unit['subj_sess'], lfp_hemroi)
        phase = dio.open_pickle(op.join(data_dir, 'spectral', 'phase', basename_lfp))
        if (phase.buffer > 0) & (not phase.clip_buffer):
            phase = phase.loc[:, :, :, phase.buffer:phase.time.size-phase.buffer-1]
            phase.attrs['clip_buffer'] = True
        lfp_chans = [chan for chan in phase.chan.values if chan not in [unit['chan']]]
        if unit['chan'] in phase.chan:
            phase = phase.loc[{'chan': lfp_chans}]
        if not np.array_equal(phase.freq.values, freqs):
            phase = phase.loc[{'freq': [freq for freq in phase.freq.values if freq in freqs]}]
        if np.nanmax(phase.values) > np.pi:
            phase -= np.pi # convert values back to -π to π, with -π being the trough.
        event_dur = phase.time.size

        # Load the P-episode mask for each event.
        roi_keys = ['comp_lfp', 'target_lfp']
        mask_keys = ['target_and_comp', 'target_not_comp']
        osc_mask = {}
        for key in roi_keys:
            hemroi = lfp_hemroi if key=='target_lfp' else comp_hemroi
            basename = '{}-{}.pkl'.format(unit['subj_sess'], hemroi)
            osc_mask[key] = dio.open_pickle(op.join(data_dir, 'p_episode', basename))
            if (osc_mask[key].buffer > 0) & (not osc_mask[key].clip_buffer):
                osc_mask[key] = osc_mask[key].loc[:, :, :, osc_mask[key].buffer:osc_mask[key].time.size-osc_mask[key].buffer-1]
                osc_mask[key].attrs['clip_buffer'] = True
            if unit['chan'] in osc_mask[key].chan:
                osc_mask[key] = osc_mask[key].loc[{'chan': [chan for chan in osc_mask[key].chan.values
                                                            if chan not in [unit['chan']]]}]
            if not np.array_equal(osc_mask[key].freq.values, freqs):
                osc_mask[key] = osc_mask[key].loc[{'freq': [freq for freq in osc_mask[key].freq.values
                                                            if freq in freqs]}]
        osc_mask['target_and_comp'] = osc_mask['target_lfp'].copy(data=osc_mask['target_lfp'].values & np.any(osc_mask['comp_lfp'].values, axis=1)[:, None, :, :])
        osc_mask['target_not_comp'] = osc_mask['target_lfp'].copy(data=osc_mask['target_lfp'].values & ~np.any(osc_mask['comp_lfp'].values, axis=1)[:, None, :, :])
        
        # Calculate P-episode at each frequency, across channels and events.
        peps = {key: 100 * osc_mask[key].mean(dim=('event', 'chan', 'time')).values
                for key in mask_keys}
        max_pep = {key: np.nanmax(peps[key]) for key in mask_keys}
        max_pep_freq = {key: np.nanargmax(peps[key]) for key in mask_keys}
        
        # Ensure that phase, oscillation mask, and spike_times are all in the same event order.
        assert np.array_equal(phase.event.values, osc_mask[key].event.values)
        assert np.array_equal(phase.event.values,
                              np.array(event_time_spikes.apply(lambda x: (x['gameState'],
                                                                          x['trial']),
                                                               axis=1).values))

        # ------------------------------------
        # Calculate real phase-locking
        
        # Get a masked array of spike phases, across events, during active
        # oscillations at each channel and frequency.
        spike_phases = {key: np.ma.concatenate(event_time_spikes.apply(
                            lambda x: get_spike_phases(
                                x['spike_times'],
                                phase.values[x['event_idx'], :, :, :],
                                mask=np.invert(osc_mask[key].values[x['event_idx'], ...]),
                                event_dur=event_dur,
                                circshift=False),
                            axis=1).tolist(), axis=-1) # chan x freq x spike
                        for key in mask_keys}

        # Match the number of spikes sampled between mask keys, at each channel and frequency.
        if match_spikes:
            min_spikes = np.min([np.count_nonzero(~spike_phases[key].mask, axis=2)
                                 for key in mask_keys], axis=0) # (chan, freq)
            for key in mask_keys:
                spike_phases[key].mask = subsample_mask(spike_phases[key].mask, size=min_spikes)

        # Calculate mean resultant lengths for each channel and frequency.
        mrls = {key: np.nanmean([[circstats.circmoment(spike_phases[key][iChan, iFreq, :].compressed())[1]
                                  for iFreq in range(spike_phases[key].shape[1])]
                                 for iChan in range(spike_phases[key].shape[0])], axis=0).astype(np.float32) # (freq,)
                for key in mask_keys}

        # Log how many spikes were counted per frequency,
        # taking the mean across channels.
        n_spikes_mask = {key: np.nanmean([[spike_phases[key][iChan, iFreq, :].flatten().compressed().size
                                           for iFreq in range(spike_phases[key].shape[1])]
                                          for iChan in range(spike_phases[key].shape[0])], axis=0).astype(np.float32) # (freq,)
                         for key in mask_keys}

        # Get the preferred phase at each frequency, across channels.
        pref_phases = {key: np.array([circstats.circmoment(spike_phases[key][:, iFreq, :].flatten().compressed())[0]
                                      for iFreq in range(spike_phases[key].shape[1])]).astype(np.float32) # (freq,)
                       for key in mask_keys}

        # ------------------------------------
        # Calculate null phase-locking
        mrls_null = {key: [] for key in mask_keys}
        n_spikes_mask_null = {key: [] for key in mask_keys}
        for iPerm in range(n_perm):
            # Get a masked array of spike phases, across events, during active
            # oscillations at each channel and frequency.
            _spike_phases_null = {key: np.ma.concatenate(event_time_spikes.apply(
                                      lambda x: get_spike_phases(
                                          x['spike_times'],
                                          phase.values[x['event_idx'], :, :, :],
                                          mask=np.invert(osc_mask[key].values[x['event_idx'], ...]),
                                          event_dur=event_dur,
                                          circshift=True),
                                      axis=1).tolist(), axis=-1) # chan x freq x spike
                                  for key in mask_keys}

            # Match the number of spikes sampled between mask keys, at each channel and frequency.
            if match_spikes:
                min_spikes = np.min([np.count_nonzero(~_spike_phases_null[key].mask, axis=2)
                                     for key in mask_keys], axis=0) # (chan, freq)
                for key in mask_keys:
                    _spike_phases_null[key].mask = subsample_mask(_spike_phases_null[key].mask, size=min_spikes)

            # Calculate mean resultant lengths for each channel and frequency.
            _mrls_null = {key: np.nanmean([[circstats.circmoment(_spike_phases_null[key][iChan, iFreq, :].compressed())[1]
                                      for iFreq in range(_spike_phases_null[key].shape[1])]
                                     for iChan in range(_spike_phases_null[key].shape[0])], axis=0).astype(np.float32) # (freq,)
                          for key in mask_keys}
            for key in mask_keys:
                mrls_null[key].append(_mrls_null[key].tolist())

            # Log how many spikes were counted per frequency,
            # taking the mean across channels.
            _n_spikes_mask_null = {key: np.nanmean([[_spike_phases_null[key][iChan, iFreq, :].flatten().compressed().size
                                                     for iFreq in range(_spike_phases_null[key].shape[1])]
                                                    for iChan in range(_spike_phases_null[key].shape[0])], axis=0).astype(np.float32) # (freq,)
                                   for key in mask_keys}
            for key in mask_keys:
                n_spikes_mask_null[key].append(_n_spikes_mask_null[key])

        mrls_null = {key: np.array(mrls_null[key]) for key in mask_keys} # (perm, freq)
        n_spikes_mask_null = {key: np.array(n_spikes_mask_null[key]) for key in mask_keys} # (perm, freq)


        # ------------------------------------
        # Statistics

        # Z-score MRLs against the null distribution,
        # and calculate empirical P-values.
        mean_mrls_null = {key: np.nanmean(mrls_null[key], axis=0) for key in mask_keys}
        std_mrls_null = {key: np.nanstd(mrls_null[key], axis=0) for key in mask_keys}
        z_mrls = {key: (mrls[key] - mean_mrls_null[key]) / std_mrls_null[key] # (freq,)
                  for key in mask_keys}
        z_mrls_null = {key: (mrls_null[key] - mean_mrls_null[key][None, :])
                            / std_mrls_null[key][None, :] # (perm, freq)
                       for key in mask_keys}
        max_z_mrl = {key: np.nanmax(z_mrls[key]) for key in mask_keys}
        max_z_mrl_freq = {key: freqs[np.nanargmax(z_mrls[key])] for key in mask_keys}
        pval = {key: (1 + np.nansum(np.nanmax(z_mrls_null[key], axis=1) >= max_z_mrl[key]))
                     / (1 + n_perm)
                for key in mask_keys}
        sig = {key: pval[key] < alpha for key in mask_keys}

        mean_n_spikes_mask_null = {key: np.nanmean(n_spikes_mask_null[key], axis=0)
                                   for key in mask_keys}
        std_n_spikes_mask_null = {key: np.nanstd(n_spikes_mask_null[key], axis=0)
                                  for key in mask_keys}
        z_n_spikes_mask = {key: (n_spikes_mask[key] - mean_n_spikes_mask_null[key])
                                / std_n_spikes_mask_null[key]
                           for key in mask_keys}

        # ------------------------------------
        # Add results to dataframe.
        for key in mask_keys:
            pl_mrls.append([unit['subj'],
                            unit['subj_sess'],
                            '{}-{}'.format(unit['chan'], unit['unit']),
                            unit['hemroi'],
                            unit['roi_gen'],
                            unit['n_spikes'],
                            unit['fr'],
                            lfp_hemroi,
                            lfp_roi_gen,
                            lfp_chans,
                            edge,
                            same_hem,
                            same_roi_gen,
                            comp_hemroi,
                            comp_roi_gen,
                            key,
                            peps[key],
                            max_pep[key],
                            max_pep_freq[key],
                            n_spikes_mask[key],
                            n_spikes_mask_null[key],
                            mrls[key],
                            mrls_null[key],
                            mean_mrls_null[key].tolist(),
                            std_mrls_null[key].tolist(),
                            pref_phases[key],
                            z_n_spikes_mask[key],
                            z_mrls[key],
                            max_z_mrl[key],
                            max_z_mrl_freq[key],
                            pval[key],
                            sig[key]])

    # Create the output dataframe.
    cols = ['subj', 'subj_sess', 'unit', 'unit_hemroi', 'unit_roi_gen', 'n_spikes', 'fr',
            'lfp_hemroi', 'lfp_roi_gen', 'lfp_chans', 'edge', 'same_hem', 'same_roi_gen',
            'comp_hemroi', 'comp_roi_gen', 'mask',
            'peps', 'max_pep', 'max_pep_freq', 'n_spikes_mask', 'n_spikes_mask_null',
            'mrls', 'mrls_null', 'mean_mrls_null', 'std_mrls_null', 'pref_phases',
            'z_n_spikes_mask', 'z_mrls', 'max_z_mrl', 'max_z_mrl_freq', 'pval', 'sig']
    pl_mrls = pd.DataFrame(pl_mrls, columns=cols)

    # Save the output.
    if save_output:
        dio.save_pickle(pl_mrls, output_f, verbose)
    
    if verbose:
        print('pl_mrls: {}'.format(pl_mrls.shape))
        print(timer)
        
    return pl_mrls


def unit_to_lfp_phase_locking(unit,
                              expmt='goldmine',
                              game_states=['Encoding', 'Retrieval'],
                              freqs=np.arange(1, 31),
                              n_rois=8,
                              keep_same_hem=[True, False],
                              keep_edges=['hpc-local', 'hpc-hpc', 'hpc-ctx', 'ctx-local', 'ctx-hpc', 'ctx-ctx'],
                              exclude_gen_rois=[],
                              mask_phase=True,
                              n_perm=1000,
                              alpha=0.05,
                              data_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/goldmine/nav',
                              output_dir=None,
                              save_output=True,
                              overwrite=False,
                              verbose=True):
    """Calculate unit's phase-locking across events, at each frequency.
    
    Parameters
    ----------
    unit : dict or series
        Contains the spike times vector and identifying info for a unit.
    expmt : str, 'goldmine' | 'ycab'
        Indicates which dataset we're working with.
    game_states : array
        Event intervals to include in the analysis.
    freqs : array
        Frequencies at which spike-phase relations are analyzed.
    n_rois : int
        Number of meta-regions to assign for categorization.
    keep_same_hem : list[bool]
        Which hemispheric connections to process. Default is to run both
        ipsilateral and contralateral.
    keep_edges : list[str]
        Define the edges to be processed. Default is to analyze all
        electrode pairs.
    exclude_gen_rois : list[str]
        Exclude LFP gen_roi regions in this list.
    mask_phase : bool
        If True, only spikes that occur during P-episode-defined
        oscillatory states are included in the analyses. This masking is
        done both for the real spike times and for circularly-shifted
        null spike times.
    n_perm : int
        Number of permutations drawn to construst the null distribution.
        For each permutation, spike times are circ-shifted at random
        within each event, and phase-locking values at each frequency
        are recalculcated across events.
    alpha : float
        Defines the significance threshold for the phase-locking
        empirical p-value.
    data_dir : str
        Filepath to the location where saved inputs are stored.
    output_dir : str | None
        Filepath to the location where the output file is saved.
    save_output : bool
        Output is saved only if True.
    overwrite : bool
        If False and saved output already exists, it is simply returned
        at the top of this function. Otherwise phase-locking is
        calculated and, if save_output is True, any existing output file
        is overwritten.
    verbose : bool
        If True, some info is printed to the standard output.
        
    Returns
    -------
    pl_mrls : dataframe
        Each row corresponds to one unit -> LFPs from one microwire
        bundle.
    """
    timer = Timer()

    # Load the output file if it exists.
    output_b = '{}-{}-{}.pkl'.format(unit['subj_sess'], unit['chan'], unit['unit'])
    if output_dir is None:
        output_dir = op.join(data_dir, 'phase_locking')
    output_f = op.join(output_dir, output_b)
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Loading saved output: {}'.format(output_f))
        return dio.open_pickle(output_f)

    # Get a list of LFP ROIs to process.
    if expmt == 'goldmine':
        mont = spike_preproc.get_montage(unit['subj_sess'])
    elif expmt == 'ycab':
        subj_df = _get_subj_df()
        subj_df['subj_sess'] = subj_df['subj_sess'].apply(
            lambda x: str_replace(x, {'env': 'ses', '1a': '1'}))
        mont = (subj_df.query("(subj_sess=='{}')".format(unit['subj_sess']))
                       .groupby(['location'])['chan']
                       .apply(lambda x: np.array([int(chan) for chan in x])))
    roi_map = spike_preproc.roi_mapping(n=n_rois)
    process_pairs = _get_process_pairs(mont,
                                       unit['hemroi'],
                                       keep_same_hem,
                                       keep_edges,
                                       exclude_gen_rois,
                                       roi_map)

    pl_mrls = []
    for (unit_hemroi, lfp_hemroi) in process_pairs:
        # ------------------------------------
        # Data loading

        # Determine which regions we're looking at.
        same_hem = (unit_hemroi[0] == lfp_hemroi[0])
        edge = _get_edge(unit_hemroi, lfp_hemroi)
        unit_roi_gen = roi_map[unit_hemroi[1:]]
        lfp_roi_gen = roi_map[lfp_hemroi[1:]]
        same_roi_gen = (unit_roi_gen == lfp_roi_gen)        

        # Load event_times.
        event_times = load_event_times(unit['subj_sess'],
                                       expmt=expmt,
                                       game_states=game_states,
                                       data_dir=data_dir)

        # Calculate spike times relative to the start of each event.
        event_time_spikes = load_event_time_spikes(event_times,
                                                   unit['spike_times'])

        # Load phase values for each event.
        basename_lfp = '{}-{}.pkl'.format(unit['subj_sess'], lfp_hemroi)
        phase = dio.open_pickle(op.join(data_dir, 'spectral', 'phase', basename_lfp))
        if (phase.buffer > 0) & (not phase.clip_buffer):
            phase = phase.loc[:, :, :, phase.buffer:phase.time.size-phase.buffer-1]
            phase.attrs['clip_buffer'] = True
        lfp_chans = [chan for chan in phase.chan.values if chan not in [unit['chan']]]
        if unit['chan'] in phase.chan:
            phase = phase.loc[{'chan': lfp_chans}]
        if not np.array_equal(phase.freq.values, freqs):
            phase = phase.loc[{'freq': [freq for freq in phase.freq.values if freq in freqs]}]
        if np.nanmax(phase.values) > np.pi:
            phase -= np.pi # convert values back to -π to π, with -π being the trough.

        # Load the P-episode mask for each event.
        osc_mask = dio.open_pickle(op.join(data_dir, 'p_episode', basename_lfp))
        if (osc_mask.buffer > 0) & (not osc_mask.clip_buffer):
            osc_mask = osc_mask.loc[:, :, :, osc_mask.buffer:osc_mask.time.size-osc_mask.buffer-1]
            osc_mask.attrs['clip_buffer'] = True
        if unit['chan'] in osc_mask.chan:
            osc_mask = osc_mask.loc[{'chan': [chan for chan in osc_mask.chan.values
                                              if chan not in [unit['chan']]]}]
        if not np.array_equal(osc_mask.freq.values, freqs):
            osc_mask = osc_mask.loc[{'freq': [freq for freq in osc_mask.freq.values
                                              if freq in freqs]}]

        # Ensure that phase, oscillation mask, and spike_times are all in the same event order.
        assert np.array_equal(phase.event.values, osc_mask.event.values)
        assert np.array_equal(phase.event.values,
                              np.array(event_time_spikes.apply(lambda x: (x['gameState'],
                                                                          x['trial']),
                                                               axis=1).values))

        # Calculate P-episode at each frequency, across channels and events.
        peps = 100 * osc_mask.mean(dim=('event', 'chan', 'time')).values
        max_pep = peps.max()
        max_pep_freq = freqs[peps.argmax()]

        # ------------------------------------
        # Calculate real phase-locking

        # Get a masked array of spike phases, across events, during active
        # oscillations at each channel and frequency.
        event_dur = phase.time.size
        if mask_phase:
            spike_phases = np.ma.concatenate(event_time_spikes.apply(
                lambda x: get_spike_phases(x['spike_times'],
                                           phase.values[x['event_idx'], :, :, :],
                                           mask=np.invert(osc_mask.values[x['event_idx'], :, :, :]),
                                           event_dur=event_dur,
                                           circshift=False),
                axis=1).tolist(), axis=-1) # chan x freq x spike
        else:
            spike_phases = np.ma.concatenate(event_time_spikes.apply(
                lambda x: get_spike_phases(x['spike_times'],
                                           phase.values[x['event_idx'], :, :, :],
                                           mask=None,
                                           event_dur=event_dur,
                                           circshift=False),
                axis=1).tolist(), axis=-1) # chan x freq x spike

        # Calculate mean resultant lengths for each channel and frequency.
        mrls = np.nanmean([[circstats.circmoment(spike_phases[iChan, iFreq, :].compressed())[1]
                            for iFreq in range(spike_phases.shape[1])]
                           for iChan in range(spike_phases.shape[0])], axis=0).astype(np.float32) # (freq,)

        # Log how many spikes were counted per frequency,
        # taking the mean across channels.
        n_spikes_mask = np.nanmean([[spike_phases[iChan, iFreq, :].flatten().compressed().size
                                     for iFreq in range(spike_phases.shape[1])]
                                    for iChan in range(spike_phases.shape[0])], axis=0).astype(np.float32) # (freq,)

        # Get the preferred phase at each frequency, across channels.
        pref_phases = np.array([circstats.circmoment(spike_phases[:, iFreq, :].flatten().compressed())[0]
                                for iFreq in range(spike_phases.shape[1])]).astype(np.float32) # (freq,)

        # ------------------------------------
        # Calculate null phase-locking
        mrls_null = []
        n_spikes_mask_null = []
        for iPerm in range(n_perm):
            # Get a masked array of spike phases, across events, during active
            # oscillations at each channel and frequency.
            if mask_phase:
                _spike_phases_null = np.ma.concatenate(event_time_spikes.apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase[x['event_idx'], :, :, :],
                                               mask=np.invert(osc_mask[x['event_idx'], :, :, :]),
                                               event_dur=event_dur,
                                               circshift=True),
                    axis=1).tolist(), axis=-1) # chan x freq x spike
            else:
                _spike_phases_null = np.ma.concatenate(event_time_spikes.apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase[x['event_idx'], :, :, :],
                                               mask=None,
                                               event_dur=event_dur,
                                               circshift=True),
                    axis=1).tolist(), axis=-1) # chan x freq x spike

            # Calculate mean resultant lengths for each channel and frequency.
            _mrls_null = np.nanmean([[circstats.circmoment(_spike_phases_null[iChan, iFreq, :].compressed())[1]
                                      for iFreq in range(_spike_phases_null.shape[1])]
                                     for iChan in range(_spike_phases_null.shape[0])], axis=0).astype(np.float32)
            mrls_null.append(_mrls_null.tolist())

            _n_spikes_mask_null = np.nanmean([[_spike_phases_null[iChan, iFreq, :].flatten().compressed().size
                                               for iFreq in range(_spike_phases_null.shape[1])]
                                              for iChan in range(_spike_phases_null.shape[0])], axis=0).astype(np.float32)
            n_spikes_mask_null.append(_n_spikes_mask_null)

        mrls_null = np.array(mrls_null) # (perm, freq)
        n_spikes_mask_null = np.array(n_spikes_mask_null) # (perm, freq)

        # ------------------------------------
        # Statistics

        # Z-score MRLs against the null distribution,
        # and calculate empirical P-values.
        mean_mrls_null = np.nanmean(mrls_null, axis=0)
        std_mrls_null = np.nanstd(mrls_null, axis=0)
        z_mrls = (mrls - mean_mrls_null) / std_mrls_null # (freq,)
        z_mrls_null = (mrls_null - mean_mrls_null[None, :]) / std_mrls_null[None, :] # (perm, freq)
        max_z_mrl = np.nanmax(z_mrls)
        max_z_mrl_freq = freqs[np.nanargmax(z_mrls)]
        pval = (1 + np.nansum(np.nanmax(z_mrls_null, axis=1) >= max_z_mrl)) / (1 + n_perm)
        sig = pval < alpha

        mean_n_spikes_mask_null = np.nanmean(n_spikes_mask_null, axis=0)
        std_n_spikes_mask_null = np.nanstd(n_spikes_mask_null, axis=0)
        z_n_spikes_mask = (n_spikes_mask - mean_n_spikes_mask_null) / std_n_spikes_mask_null

        # ------------------------------------
        # Add results to dataframe.
        pl_mrls.append([unit['subj'],
                        unit['subj_sess'],
                        '{}-{}'.format(unit['chan'], unit['unit']),
                        unit_hemroi,
                        unit_roi_gen,
                        unit['n_spikes'],
                        unit['fr'],
                        lfp_hemroi,
                        lfp_roi_gen,
                        lfp_chans,
                        edge,
                        same_hem,
                        same_roi_gen,
                        peps,
                        max_pep,
                        max_pep_freq,
                        n_spikes_mask,
                        n_spikes_mask_null,
                        mrls,
                        mrls_null,
                        mean_mrls_null.tolist(),
                        std_mrls_null.tolist(),
                        pref_phases,
                        z_n_spikes_mask,
                        z_mrls,
                        max_z_mrl,
                        max_z_mrl_freq,
                        pval,
                        sig])

    # Create the output dataframe.
    cols = ['subj', 'subj_sess', 'unit', 'unit_hemroi', 'unit_roi_gen', 'n_spikes', 'fr',
            'lfp_hemroi', 'lfp_roi_gen', 'lfp_chans', 'edge', 'same_hem', 'same_roi_gen',
            'peps', 'max_pep', 'max_pep_freq', 'n_spikes_mask', 'n_spikes_mask_null',
            'mrls', 'mrls_null', 'mean_mrls_null', 'std_mrls_null', 'pref_phases',
            'z_n_spikes_mask', 'z_mrls', 'max_z_mrl', 'max_z_mrl_freq', 'pval', 'sig']
    pl_mrls = pd.DataFrame(pl_mrls, columns=cols)

    # Save the output.
    if save_output:
        dio.save_pickle(pl_mrls, output_f, verbose)
    
    if verbose:
        print('pl_mrls: {}'.format(pl_mrls.shape))
        print(timer)
        
    return pl_mrls


def _get_process_pairs(mont,
                       unit_hemroi,
                       keep_same_hem=[True, False],
                       keep_edges=['hpc-local', 'hpc-hpc', 'hpc-ctx', 'ctx-local', 'ctx-hpc', 'ctx-ctx'],
                       exclude_gen_rois=[],
                       roi_map=None):
    """Return a list of (unit, lfp) region pairs to process."""
    hpc_rois = ['AH', 'MH', 'PH']
    hemroi_pairs = [(unit_hemroi, lfp_hemroi) for lfp_hemroi in sorted(mont.keys())]
    process_pairs = []
    for (unit_hemroi, lfp_hemroi) in hemroi_pairs:
        same_hem = (unit_hemroi[0] == lfp_hemroi[0])
        is_local = (unit_hemroi == lfp_hemroi)
        unit_is_hpc = (unit_hemroi[1:] in hpc_rois)
        lfp_is_hpc = (lfp_hemroi[1:] in hpc_rois)
        edge = _get_edge(unit_hemroi, lfp_hemroi)

        exclude_lfp_hemroi = False
        if np.all((len(exclude_gen_rois) > 0,
                   roi_map is not None)):
            lfp_roi_gen = roi_map[lfp_hemroi[1:]]
            if lfp_roi_gen in exclude_gen_rois:
                exclude_lfp_hemroi = True
        
        if np.all((same_hem in keep_same_hem,
                   edge in keep_edges,
                   not exclude_lfp_hemroi)):
            process_pairs.append([unit_hemroi, lfp_hemroi])
    return process_pairs


def _get_edge(unit_hemroi, lfp_hemroi):
    """Return the edge type."""
    hpc_rois = ['AH', 'MH', 'PH']
    
    same_hem = (unit_hemroi[0] == lfp_hemroi[0])
    is_local = (unit_hemroi == lfp_hemroi)
    unit_is_hpc = (unit_hemroi[1:] in hpc_rois)
    lfp_is_hpc = (lfp_hemroi[1:] in hpc_rois)
    
    if np.all((unit_is_hpc, is_local)):
        edge = 'hpc-local'
    elif np.all((unit_is_hpc, lfp_is_hpc, not is_local)):
        edge = 'hpc-hpc'
    elif np.all((unit_is_hpc, not lfp_is_hpc)):
        edge = 'hpc-ctx'
    elif np.all((not unit_is_hpc, is_local)):
        edge = 'ctx-local'
    elif np.all((not unit_is_hpc, lfp_is_hpc)):
        edge = 'ctx-hpc'
    elif np.all((not unit_is_hpc, not lfp_is_hpc, not is_local)):
        edge = 'ctx-ctx'
    
    return edge


def axe_connections(_pl_mrls):
    """Return a list of indices to remove.
    
    Drops all but one hippocampal connection to each hemisphere.
    Where possible, keeps connections to hippocampal electrodes
    that are located at similar longitudinal positions in both
    hemispheres. Otherwise, keeps connections to the most
    posterior hippocampal electrode in each hemisphere.
    """
    ap_order = ['P', 'M', 'A']
    n_edges = _pl_mrls['lfp_hemroi'].apply(lambda x: x[1]).value_counts().to_dict()
    for ap in ap_order:
        if n_edges.get(ap, 0) > 1:
            keep_rois = ['L{}H'.format(ap), 'R{}H'.format(ap)]
            drop_idx = _pl_mrls.query("(lfp_hemroi!={})".format(keep_rois)).index.tolist()
            return drop_idx
    keep_rois = []
    for hem in ['L', 'R']:
        for ap in ap_order:
            test_roi = '{}{}H'.format(hem, ap)
            if test_roi in _pl_mrls['lfp_hemroi'].tolist():
                keep_rois.append(test_roi)
                break
    drop_idx = _pl_mrls.query("(lfp_hemroi!={})".format(keep_rois)).index.tolist()
    return drop_idx


def subsample_mask(mask_in, size):
    """Return a randomly subsampled mask, without replacement.
    
    Samples are taken over the last axis. mask_in must be
    3-dimensional, envisioned as chan x freq x spike but
    technically it doesn't *have* to be like this.
    
    Parameters
    ----------
    mask_in : array, shape=(chan, freq, spike), dype=bool
        True = masked values, False = unmasked values.
    size : array, shape=(chan, freq)
        How many unmasked to subsample, without replacement,
        at each channel and frequency.
        
    Returns
    -------
    mask_out : array, shape=mask_in.shape, dtype=bool
        Note if size == np.count_nonzero(~mask_in, axis=2),
        the original mask is returned.
    """
    shp = mask_in.shape
    unmask_in = np.invert(mask_in)
    mask_out = np.ones(mask_in.shape, dtype=bool)
    for ii in range(shp[0]):
        for jj in range(shp[1]):
            kk = np.random.choice(np.where(unmask_in[ii, jj, :])[0],
                                  size=size[ii, jj],
                                  replace=False)
            mask_out[ii, jj, kk] = False
    
    return mask_out


def load_event_times(subj_sess,
                     expmt='goldmine',
                     game_states=['Encoding', 'Retrieval'],
                     data_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/goldmine/nav'):
    """Return the event_times dataframe.
    
    Parameters
    ----------
    expmt : str, 'goldmine' | 'ycab'
        Indicates which dataset we're working with.
    game_states : array
        Event intervals to include in the analysis.

    Returns
    -------
    event_times : dataframe
        Each row = one event interval.
    """
    if expmt == 'goldmine':
        event_times = (events_proc.load_events(subj_sess, verbose=0)
                                  .event_times
                                  .query("(gameState=={})".format(game_states))
                                  .reset_index(drop=True))
        event_times['start_time'] = event_times['time_bins'].apply(lambda x: x[0])
        event_times['stop_time'] = event_times['time_bins'].apply(lambda x: x[-1])
        event_times = event_times.sort_values(['gameState', 'trial']).reset_index(drop=True)
    elif expmt == 'ycab':
        filename = op.join(data_dir, 'events', '{}-event_times.pkl'.format(subj_sess))
        event_times = dio.open_pickle(filename)

    return event_times


def load_event_time_spikes(event_times,
                           spike_times):
    """Return vectors of spike times within each event.

    Spike times are returned relative to the start of each event,
    not the start of the session.
    
    Parameters
    ----------
    event_times : dataframe
        Each row = one event interval.
    spike_times : array
        Vector of spike times relative to session start.

    Output
    ------
    event_time_spikes : dataframe
        Same as the inputted event_times dataframe, but with a
        'spike_times' column added.
    """
    event_time_spikes = event_times[['trial', 'gameState', 'start_time', 'stop_time']].copy()
    event_time_spikes['spike_times'] = event_time_spikes.apply(
        lambda x: retain_spikes(spike_times, x['start_time'], x['stop_time']), axis=1)
    event_time_spikes = event_time_spikes.reset_index().rename(columns={'index': 'event_idx'})
    
    return event_time_spikes


def retain_spikes(spike_times,
                  start,
                  stop,
                  subtract_start=True):
    """Return spike times >= start and < stop."""
    idx = np.where((spike_times >= start) & (spike_times < stop))[0]
    if subtract_start:
        return spike_times[idx] - start
    else:
        return spike_times[idx]


def get_spike_phases(spike_times,
                     phase,
                     mask=None,
                     circshift=False,
                     event_dur=30000):
    """Return spike phases for a selected event.
    
    Parameters
    ----------
    spike_times : array, shape=(spike,)
        Spike times relative to event onset.
    phase : array, shape=(chan, freq, time)
        Phase values for the event.
    mask : array, shape=(chan, freq, time)
        Mask of timepoints to exclude from the analysis.
    circshift : bool
        If True, circ-shifts spike times by a random number
        between 0 and the event duration - 1.
    event_dur : int
        The event duration, in ms.
        
    Returns
    -------
    spike_phases : MaskedArray, shape=(chan, freq, time)
        The phase at each spike time, for each channel and
        frequency. 
    """
    if circshift:
        spike_train = np.zeros(event_dur, dtype=bool)
        spike_train[spike_times] = True
        roll_by = np.random.randint(0, len(spike_train))
        spike_train = np.roll(spike_train, roll_by)
        spike_times = np.where(spike_train)[0]
                              
    spike_phases = np.ma.array(phase[:, :, spike_times]) # chan x freq x spike
    
    if mask is not None:
        spike_phases.mask = mask[:, :, spike_times]
    
    return spike_phases


def load_all_unit_spikes(fr_thresh=0.2,
                         nspike_thresh=400,
                         expmt='goldmine',
                         game_states=['Encoding', 'Retrieval'],
                         n_rois=8,
                         sessions=None,
                         verbose=True):
    """Load spike times for all units across subjects and sessions."""
    def _get_spike_times(subj_sess,
                         unit,
                         sr_saved=2000,
                         sr_out=1000,
                         spike_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data/crosselec_phase_locking/spike_inds2',
                         event_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/ycab/events'):
        sr_ratio = sr_out / sr_saved
        spike_fname = op.join(spike_dir, 'spike_inds-{}Hz-{}-unit{}.pkl'.format(sr_saved, subj_sess, unit))
        spike_times = np.unique(dio.open_pickle(spike_fname)[0] * sr_ratio).astype(int)
        event_times = dio.open_pickle(op.join(event_dir, '{}-event_times.pkl'.format(subj_sess)))
        sess_start = event_times['start_time'].min()
        sess_stop = event_times['stop_time'].max()
        spike_times = retain_spikes(spike_times, sess_start, sess_stop, subtract_start=False)
        return spike_times

    def _calc_fr(subj_sess,
                 spike_times,
                 event_dir='/home1/dscho/projects/unit_activity_and_hpc_theta/data2/ycab/events'):
        event_times = dio.open_pickle(op.join(event_dir, '{}-event_times.pkl'.format(subj_sess)))
        sess_start = event_times['start_time'].min()
        sess_stop = event_times['stop_time'].max()
        dur = (sess_stop - sess_start) * 1e-3
        fr = len(spike_times) / dur
        return fr

    timer = Timer()
    
    roi_map = spike_preproc.roi_mapping(n_rois)
    if expmt == 'goldmine':
        # Load all unit spike times.
        cols = ['subj', 'subj_sess', 'chan', 'unit',
                'hemroi', 'n_spikes', 'fr', 'spike_times']
        if sessions is None:
            sessions = np.unique([op.basename(f).split('-')[0]
                                  for f in glob(op.join('/data7/goldmine/analysis/events', '*.pkl'))])
            if verbose:
                print('{} subjects, {} sessions'
                      .format(len(np.unique([x.split('_')[0] for x in sessions])), len(sessions)))

        spikes = []
        for subj_sess in sessions:
            evsp = time_bin_analysis.load_event_spikes(subj_sess, verbose=0)
            neurons = evsp.column_map['neurons']
            spikes += [spike_preproc.load_spikes(subj_sess, neuron)
                       for neuron in neurons]
        spikes = pd.concat(spikes, axis=1).T[cols]

        # Add ROI info.
        spikes['hemroi'] = spikes.apply(
            lambda x: spike_preproc.roi_lookup(x['subj_sess'], x['chan']), axis=1)
        spikes.insert(spikes.columns.tolist().index('hemroi')+1,
                      'hem',
                      spikes['hemroi'].apply(lambda x: x[0]))
        spikes.insert(spikes.columns.tolist().index('hemroi')+2,
                      'roi_gen',
                      spikes['hemroi'].apply(lambda x: roi_map[x[1:]]))

        # Calculate the number of spikes, and firing rate,
        # of each unit during the time intervals of interest.
        for subj_sess in sessions:
            events = events_proc.load_events(subj_sess, verbose=0)
            events.event_times = (events.event_times
                                  .query("(gameState=={})".format(game_states))
                                  .reset_index(drop=True))
            events.event_times['start_time'] = events.event_times['time_bins'].apply(lambda x: x[0])
            events.event_times['stop_time'] = events.event_times['time_bins'].apply(lambda x: x[-1])
            idx = spikes.query("(subj_sess=='{}')".format(subj_sess)).index.tolist()
            spikes.loc[idx, 'n_spikes'] = spikes.loc[idx, 'spike_times'].apply(
                lambda spike_times: np.sum(time_bin_analysis.spikes_per_timebin(events.event_times,
                                                                                spike_times)))
            spikes.loc[idx, 'fr'] = spikes.loc[idx, 'n_spikes'].apply(
                lambda x: x / (events.event_times['time_bin_dur'].sum() * 1e-3))
    elif expmt == 'ycab':
        cols = ['subj', 'subj_sess', 'unit_chan_ind', 'unit', 'unit_hemroi']
        if sessions is None:
            globstr = op.join('/home1/dscho/projects/unit_activity_and_hpc_theta/data2/ycab/events',
                              '*-event_times.pkl')
            sessions = np.sort([op.basename(filename).split('-')[0]
                                for filename in glob(globstr)])

        spikes = _load_pl_df()
        spikes = spikes.loc[:, cols]
        spikes['subj_sess'] = spikes['subj_sess'].apply(
            lambda x: str_replace(x, {'env': 'ses', '1a': '1'}))
        spikes.insert(2, 'chan', (spikes['unit_chan_ind'] + 1).values)
        spikes.drop(columns=['unit_chan_ind'], inplace=True)
        spikes.rename(columns={'unit_hemroi': 'hemroi'}, inplace=True)
        spikes['hem'] = spikes['hemroi'].apply(lambda x: x[0])
        roi_map = spike_preproc.roi_mapping(n_rois)
        spikes['roi_gen'] = spikes['hemroi'].apply(lambda x: roi_map[x[1:]])
        spikes = spikes.drop_duplicates(['subj_sess', 'unit'])
        spikes = spikes.query("(subj_sess=={})".format(sessions.tolist()))
        spikes = spikes.sort_values(['subj_sess', 'unit']).reset_index(drop=True)

        # Load and format spike times.
        spikes['spike_times'] = spikes.apply(
            lambda x: _get_spike_times(x['subj_sess'], x['unit']), axis=1)
        spikes.insert(spikes.columns.tolist().index('roi_gen')+1,
                      'n_spikes',
                      spikes['spike_times'].apply(lambda x: len(x)))
        spikes.insert(spikes.columns.tolist().index('roi_gen')+2,
                      'fr',
                      spikes.apply(lambda x: _calc_fr(x['subj_sess'], x['spike_times']), axis=1))

    # Remove units that don't have enough spikes.
    spikes = (spikes
              .query("(fr>{}) & (n_spikes>{})".format(fr_thresh, nspike_thresh))
              .reset_index(drop=True))
    
    if verbose:
        print('spikes:', spikes.shape)
        print(timer)
        
    return spikes


def _get_subj_df(input_dir='/data3/scratch/dscho/frLfp/data/metadata'):
    """Return subj_df."""
    files = glob(op.join(input_dir, 'subj_df_*.xlsx'))
    subj_df = pd.read_excel(files[0], converters={'chan': str})
    for f in files[1:]:
        subj_df = subj_df.append(pd.read_excel(f, converters={'chan': str}))
    subj_df = subj_df.loc[subj_df.location!='none']
    return subj_df


def _load_pl_df(input_files=op.join('/home1/dscho/projects/unit_activity_and_hpc_theta/data/crosselec_phase_locking/phase_locking/unit_to_region',
                                   'all_phase_locking_stats-14026_unit_to_region_pairs-2000Hz-notch60_120Hz-5cycles-16log10freqs_0.5_to_90.5Hz.pkl'),
               drop_repeat_connections=True,
               keep_only_ctx_hpc=False):
    """Return the unit-to-region phase-locking DataFrame."""
    
    # Load the phase-locking files.
    if isinstance(input_files, str):
        pl_df = dio.open_pickle(input_files)
    else:
        pl_df = pd.DataFrame(dio.open_pickle(input_files[0])).T
        for f in input_files[1:]:
            pl_df = pl_df.append(dio.open_pickle(f))
        pl_df.reset_index(drop=True, inplace=True)

    # Ensure all columns are stored as the correct data type.
    map_dtypes = {'unit_nspikes': np.uint32,
                  'unit_fr': np.float32,
                  'lfp_is_hpc': np.bool,
                  'same_chan': np.bool,
                  'same_hemroi': np.bool,
                  'same_hem': np.bool,
                  'same_roi': np.bool,
                  'both_hpc': np.bool,
                  'same_roi2': np.bool,
                  'locked_freq_ind_z': np.uint8,
                  'locked_mrl_z': np.float64,
                  'bs_ind_z': np.uint16,
                  'bs_pval_z': np.float64,
                  'sig_z': np.bool,
                  'tl_locked_freq_z': np.uint8,
                  'tl_locked_time_z': np.int32,
                  'tl_locked_mrl_z': np.float64,
                  'pref_phase': np.float64,
                  'pref_phase_tl_locked_time_freq_z': np.float64}
    for col, dtype in map_dtypes.items():
        pl_df[col] = pl_df[col].astype(dtype)
        
    # Drop edges other than ctx-hpc.
    if keep_only_ctx_hpc:
        pl_df = pl_df.loc[pl_df['edge']=='ctx-hpc']
        
    # Add some columns to the phase-locking dataframe.
    def get_session_number(x):
        d_ = {'U380_ses1a': 1,
              'U393_ses2': 1,
              'U394_ses3': 1,
              'U396_ses2': 1,
              'U396_ses3': 2}
        if x in d_.keys():
            return d_[x]
        else:
            return int(x[-1])
    pl_df.insert(0, 'subj', pl_df.subj_sess.apply(lambda x: x.split('_')[0]))
    pl_df.insert(1, 'sess', pl_df['subj_sess'].apply(lambda x: get_session_number(x)))
    pl_df.insert(4, 'subj_unit_chan', pl_df.apply(lambda x: x['subj'] + '_' + str(x['unit_chan_ind'] + 1), axis=1))
    pl_df['unit_roi3'] = ''
    pl_df.loc[pl_df['unit_roi2'] == 'hpc', 'unit_roi3'] = 'hpc'
    pl_df.loc[pl_df['unit_roi2'] == 'ec', 'unit_roi3'] = 'ec'
    pl_df.loc[pl_df['unit_roi2'] == 'amy', 'unit_roi3'] = 'amy'
    pl_df.loc[pl_df['unit_roi3'] == '', 'unit_roi3'] = 'ctx'
    pl_df['roi'] = pl_df['unit_roi3']
    pl_df.loc[pl_df['same_hem']==False, 'roi'] = 'contra'
    roi_cats = [roi for roi in ['hpc', 'ec', 'amy', 'ctx', 'contra'] if roi in pl_df['roi'].unique()]
    pl_df['roi'] = pl_df['roi'].astype('category').cat.reorder_categories(roi_cats, ordered=True)
    pl_df['roi_unit_to_lfp'] = pl_df.apply(lambda x: x['unit_roi3'] + '_ipsi' if x['same_hem'] else x['unit_roi3'] + '_cont', axis=1)
    
    time_win = 2
    sampling_rate = 2000
    time_steps = np.arange(-time_win*sampling_rate, time_win*sampling_rate+1, sampling_rate*0.01, dtype=int)
    time_steps_ms = (time_steps / sampling_rate) * 1000
    pl_df['pl_freq'] = pl_df['locked_freq_ind_z']
    pl_df['pl_strength'] = pl_df.apply(lambda x: np.max(x['tl_mrls_z'][x['pl_freq'], :]), axis=1)
    pl_df['pl_time_shift'] = pl_df.apply(lambda x: time_steps_ms[np.argmax(x['tl_mrls_z'][x['pl_freq'], :])], axis=1)
    pl_df['pl_latency'] = np.abs(pl_df['pl_time_shift'])
    
    # Remove bad HPC electrodes (U387 sessions RAH; U394_ses3 RAH).
    pl_df.drop(index=pl_df.query("(subj_sess==['U394_ses3', 'U387_ses1', 'U387_ses2', 'U387_ses3']) & (lfp_hemroi=='RAH')").index, inplace=True)
    
    # Ensure that each neuron is compared at most once to hippocampal LFPs from
    # each hemisphere. In cases with multiple comparisons to a single
    # hemisphere, we remove all but the most posterior connection.
    def axe_connecs(x):
        """Return an empty string or a list of regions to remove."""
        removal_order = ['A', 'M', 'P']

        # Get a list of left and right ROIs, respectively.
        counts = {'left': [x_ for x_ in x if x_[0]=='L'],
                  'right': [x_ for x_ in x if x_[0]=='R']}

        # Return an empty string if neither hemisphere has more
        # than one ROI
        if np.all(np.array([len(x_) for x_ in counts.values()])<2):  
            return ''

        # For each hemisphere with more than one ROI,
        # remove all but the most posterior ROI.
        to_remove = []
        for key in counts.keys():
            vals = counts[key]
            if len(vals) > 1:
                iKeep = np.argmax([removal_order.index(val[1]) for val in vals])
                to_remove += [val for iVal, val in enumerate(vals) if iVal != iKeep]
        return to_remove
    
    if drop_repeat_connections:
        df = (pl_df
              .query("(edge=='ctx-hpc')")
              .reset_index()
              .groupby('subj_sess_unit')
              .agg({'unit': len, 
                    'lfp_hemroi': lambda x: list(x)}) 
              .query("(unit>=2)"))

        df['remove_rois'] = df['lfp_hemroi'].apply(lambda x: axe_connecs(x))

        to_remove = []
        for subj_sess_unit, row in df.iterrows():
            remove_rois = row['remove_rois']
            if len(remove_rois):
                for roi in remove_rois:
                    to_remove.append((subj_sess_unit, roi))

        remove_inds = []
        for subj_sess_unit, lfp_hemroi in to_remove:
            remove_inds.append(pl_df.query("(subj_sess_unit=='{}') & (lfp_hemroi=='{}')".format(subj_sess_unit, lfp_hemroi)).iloc[0].name)

        pl_df.drop(index=remove_inds, inplace=True)
        pl_df.reset_index(drop=True, inplace=True)
    
    # Perform FDR correction (separately for ipsilateral and contralateral
    # comparisons within each edge type).
    alpha = 0.05
    pl_df['sig_z'] = pl_df['bs_pval_z'] < alpha
    pl_df['sig_z_fdr'] = False
    for edge_type in np.unique(pl_df.edge):
        for same_hem in [True, False]:
            pvals_in = np.array(pl_df.loc[(pl_df.edge==edge_type) & (pl_df.same_hem==same_hem)].bs_pval_z.tolist())
            if len(pvals_in) > 0:
                output = multipletests(pvals_in, alpha=0.05, method='fdr_tsbky', is_sorted=False, returnsorted=False)
                sig_out = list(output[0])
                pl_df.loc[(pl_df.edge==edge_type) & (pl_df.same_hem==same_hem), 'sig_z_fdr'] = sig_out

    return pl_df
