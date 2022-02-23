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
2/23/22
"""
import sys
import os.path as op
import mkl
mkl.set_num_threads(1)
import numpy as np
import pandas as pd
import astropy.stats.circstats as circstats
sys.path.append('/home1/dscho/code/general')
import data_io as dio
from helper_funcs import Timer
sys.path.append('/home1/dscho/code/projects')
from time_cells import spike_preproc, events_proc


def unit_to_lfp_phase_locking(unit,
                              game_states=['Encoding', 'Retrieval'],
                              freqs=np.arange(1, 31),
                              n_rois=5,
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
        Contains spike times vector and identifyng info for a given neuron.
    game_states : array
        The Goldmine events for which spikes are considered.
    freqs : array
        Frequencies at which spike-phase relations are analyzed.
    n_rois : int
        Number of meta-regions to assign for categorization.
    mask_phase : bool
        If True, only spikes that occur during P-episode-defined oscillatory
        states are included in the analyses. This masking is done both for
        the real spike times and for circularly-shifted null spike times.
    n_perm : int
        Number of permutations drawn to construst the null distribution.
        For each permutation, spike times are circ-shifted at random within
        each event, and phase-locking values at each frequency are recalculcated
        across events.
    alpha : float
        Defines the significance threshold for the phase-locking empirical p-value.
    data_dir : str
        Filepath to the location where saved inputs are stored.
    output_dir : str | None
        Filepath to the location where the output file is saved.
    save_output : bool
        Output is saved only if True.
    overwrite : bool
        If False and saved output already exists, it is simply returned at the top
        of this function. Otherwise phase-locking is calculated and, if save_output
        is True, any existing output file is overwritten.
    verbose : bool
        If True, some info is printed to the standard output.
        
    Returns
    -------
    pl_mrls : dataframe
        Each row corresponds to one unit -> LFPs from one microwire bundle.
    """
    timer = Timer()

    # Load the output file if it exists.
    basename = '{}-{}-{}.pkl'.format(unit['subj_sess'], unit['chan'], unit['unit'])
    if output_dir is None:
        output_dir = op.join(data_dir, 'phase_locking')
    output_f = op.join(output_dir, basename)
    if op.exists(output_f) and not overwrite:
        if verbose:
            print('Loading saved output: {}'.format(output_f))
        return dio.open_pickle(output_f)

    # Process the local LFP and all hippocampal ROIs.
    roi_map = spike_preproc.roi_mapping(n=n_rois)
    mont = spike_preproc.get_montage(unit['subj_sess'])
    lfp_hemrois = np.unique([unit['hemroi']] + [roi for roi in mont.keys() if roi.endswith('H')])
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
        event_times = (events_proc.load_events(unit['subj_sess'], verbose=0)
                                  .event_times
                                  .query("(gameState=={})".format(game_states))
                                  .reset_index(drop=True))
        event_times['start_time'] = event_times['time_bins'].apply(lambda x: x[0])
        event_times['stop_time'] = event_times['time_bins'].apply(lambda x: x[-1])
        event_times = event_times.sort_values(['gameState', 'trial']).reset_index(drop=True)

        # Load phase values for each event.
        basename = '{}-{}.pkl'.format(unit['subj_sess'], lfp_hemroi)
        phase = dio.open_pickle(op.join(data_dir, 'spectral', 'phase', basename))
        if (phase.buffer > 0) & (not phase.clip_buffer):
            phase = phase.loc[:, :, :, phase.buffer:phase.time.size-phase.buffer-1]
            phase.attrs['clip_buffer'] = True
        lfp_chans = [chan for chan in phase.chan.values if chan not in [unit['chan']]]
        if unit['chan'] in phase.chan:
            phase = phase.loc[{'chan': lfp_chans}]
        if not np.array_equal(phase.freq.values, freqs):
            phase = phase.loc[{'freq': [freq for freq in phase.freq.values if freq in freqs]}]
        if phase.values.max() > np.pi:
            phase -= np.pi # convert values back to -π to π, with -π being the trough.

        # Load the P-episode mask for each event,
        osc_mask = dio.open_pickle(op.join(data_dir, 'p_episode', basename))
        if (osc_mask.buffer > 0) & (not osc_mask.clip_buffer):
            osc_mask = osc_mask.loc[:, :, :, osc_mask.buffer:osc_mask.time.size-osc_mask.buffer-1]
            osc_mask.attrs['clip_buffer'] = True
        if unit['chan'] in osc_mask.chan:
            osc_mask = osc_mask.loc[{'chan': [chan for chan in osc_mask.chan.values
                                              if chan not in [unit['chan']]]}]
        if not np.array_equal(osc_mask.freq.values, freqs):
            osc_mask = osc_mask.loc[{'freq': [freq for freq in osc_mask.freq.values
                                              if freq in freqs]}]

        # Calculate spike times relative to the start of each event.
        event_time_spikes = event_times[['trial', 'gameState', 'start_time', 'stop_time']].copy()
        event_time_spikes['spike_times'] = event_time_spikes.apply(
            lambda x: retain_spikes(unit['spike_times'],
                                    x['start_time'],
                                    x['stop_time']),
            axis=1)
        event_time_spikes = event_time_spikes.reset_index().rename(columns={'index': 'event_idx'})

        # Calculate P-episode at each frequency, across channels and events.
        peps = 100 * osc_mask.mean(dim=('event', 'chan', 'time')).values
        max_pep = peps.max()
        max_pep_freq = peps.argmax()

        # Ensure that phase, oscillation mask, and spike_times are all in the same event order.
        assert np.array_equal(phase.event.values, osc_mask.event.values)
        assert np.array_equal(phase.event.values,
                              np.array(event_time_spikes.apply(lambda x: (x['gameState'],
                                                                          x['trial']),
                                                               axis=1).values))

        # ------------------------------------
        # Calculate real phase-locking

        # Get a masked array of spike phases, across events, during active
        # oscillations at each channel and frequency.
        event_dur = phase.time.size
        if mask_phase:
            spike_phases = np.ma.concatenate(event_time_spikes.apply(
                lambda x: get_spike_phases(x['spike_times'],
                                           phase.values[x['event_idx'], :, :, :],
                                           osc_mask=osc_mask.values[x['event_idx'], :, :, :],
                                           event_dur=event_dur,
                                           circshift=False),
                axis=1).tolist(), axis=-1) # chan x freq x spike
        else:
            spike_phases = np.ma.concatenate(event_time_spikes.apply(
                lambda x: get_spike_phases(x['spike_times'],
                                           phase.values[x['event_idx'], :, :, :],
                                           osc_mask=None,
                                           event_dur=event_dur,
                                           circshift=False),
                axis=1).tolist(), axis=-1) # chan x freq x spike

        # Calculate mean resultant lengths for each channel and frequency.
        mrls = np.mean([[circstats.circmoment(spike_phases[iChan, iFreq, :].compressed())[1]
                         for iFreq in range(spike_phases.shape[1])]
                        for iChan in range(spike_phases.shape[0])], axis=0).astype(np.float32) # (freq,)

        # Log how many spikes were counted per frequency,
        # taking the mean across channels.
        n_spikes_mask = np.mean([[spike_phases[iChan, iFreq, :].flatten().compressed().size
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
                                               osc_mask=osc_mask[x['event_idx'], :, :, :],
                                               event_dur=event_dur,
                                               circshift=True),
                    axis=1).tolist(), axis=-1) # chan x freq x spike
            else:
                _spike_phases_null = np.ma.concatenate(event_time_spikes.apply(
                    lambda x: get_spike_phases(x['spike_times'],
                                               phase[x['event_idx'], :, :, :],
                                               osc_mask=None,
                                               event_dur=event_dur,
                                               circshift=True),
                    axis=1).tolist(), axis=-1) # chan x freq x spike

            # Calculate mean resultant lengths for each channel and frequency.
            _mrls_null = np.mean([[circstats.circmoment(_spike_phases_null[iChan, iFreq, :].compressed())[1]
                                   for iFreq in range(_spike_phases_null.shape[1])]
                                  for iChan in range(_spike_phases_null.shape[0])], axis=0).astype(np.float32)
            mrls_null.append(_mrls_null.tolist())

            _n_spikes_mask_null = np.mean([[_spike_phases_null[iChan, iFreq, :].flatten().compressed().size
                                            for iFreq in range(_spike_phases_null.shape[1])]
                                           for iChan in range(_spike_phases_null.shape[0])], axis=0).astype(np.float32)
            n_spikes_mask_null.append(_n_spikes_mask_null)

        mrls_null = np.array(mrls_null) # (perm, freq)
        n_spikes_mask_null = np.array(n_spikes_mask_null) # (perm, freq)

        # ------------------------------------
        # Statistics

        # Z-score MRLs against the null distribution,
        # and calculate empirical P-values.
        mean_mrls_null = np.mean(mrls_null, axis=0)
        std_mrls_null = np.std(mrls_null, axis=0)
        z_mrls = (mrls - mean_mrls_null) / std_mrls_null # (freq,)
        z_mrls_null = (mrls_null - mean_mrls_null[None, :]) / std_mrls_null[None, :] # (perm, freq)
        max_z_mrl = z_mrls.max()
        max_z_mrl_freq = freqs[z_mrls.argmax()]
        pval = (1 + np.sum(np.max(z_mrls_null, axis=1) >= max_z_mrl)) / (1 + n_perm)
        sig = pval < alpha

        mean_n_spikes_mask_null = np.mean(n_spikes_mask_null, axis=0)
        std_n_spikes_mask_null = np.std(n_spikes_mask_null, axis=0)
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
            'peps', 'max_pep', 'max_pep_freq', 'n_spikes_mask', 'n_spikes_mask_null',
            'mrls', 'mrls_null', 'pref_phases',
            'z_n_spikes_mask', 'z_mrls', 'max_z_mrl', 'max_z_mrl_freq', 'pval', 'sig']
    pl_mrls = pd.DataFrame(pl_mrls, columns=cols)

    # Save the output.
    if save_output:
        dio.save_pickle(osc_mask, output_f, verbose)
    
    if verbose:
        print('pl_mrls: {}'.format(pl_mrls.shape))
        print(timer)
        
    return pl_mrls


def retain_spikes(spike_times, start, stop):
    """Return spike times >= start and < stop."""
    idx = np.where((spike_times >= start) & (spike_times < stop))[0]
    return spike_times[idx] - start


def get_spike_phases(spike_times,
                     phase,
                     osc_mask=None,
                     circshift=False,
                     event_dur=30000):
    """Return spike phases for a selected event.
    
    Parameters
    ----------
    spike_times : array, shape=(spike,)
        Spike times relative to event onset.
    phase : array, shape=(chan, freq, time)
        Phase values for the event.
    osc_mask : array, shape=(chan, freq, time)
        Mask of times that an oscillation was present at
        each frequency.
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
    
    if osc_mask is not None:
        spike_phases.mask = np.invert(osc_mask[:, :, spike_times])
    
    return spike_phases
