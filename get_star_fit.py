def get_star_fit(subj):
    print subj.filts_to_use
    nuFnu_full_use = subj.nuFnu_to_use
    nuFnu_full_err_use = subj.nuFnuerrs_to_use
    mags_full_use = subj.mags_to_use
    magerrs_full_use = subj.magerrs_to_use
    cent_wavs_full_use = subj.centwavs_meters_to_use
    filterzps_full_use = subj.filterzps_to_use
            
    filter_cut = None
    
    if len(nuFnu_full_use) > 8.:
        nuFnu_use = nuFnu_full_use[:-6]
        nuFnu_err_use = nuFnu_full_err_use[:-6]
        mags_use = mags_full_use[:-6]
        magerrs_use = magerrs_full_use[:-6]
        cent_wavs_use = cent_wavs_full_use[:-6]
        filterzps_use = filterzps_full_use[:-6]
        filter_cut = -6
        
    elif len(nuFnu_full_use) > 7.:
        nuFnu_use = nuFnu_full_use[:-5]
        nuFnu_err_use = nuFnu_full_err_use[:-5]
        mags_use = mags_full_use[:-5]
        magerrs_use = magerrs_full_use[:-5]
        cent_wavs_use = cent_wavs_full_use[:-5]
        filterzps_use = filterzps_full_use[:-5]
        filter_cut = -5

    else:
        nuFnu_use = nuFnu_full_use[:-4]
        nuFnu_err_use = nuFnu_full_err_use[:-4]
        mags_use = mags_full_use[:-4]
        magerrs_use = magerrs_full_use[:-4]
        cent_wavs_use = cent_wavs_full_use[:-4]
        filterzps_use = filterzps_full_use[:-4]
        filter_cut = -4
    
    if mags_full_use[-7] - mags_full_use[-6] > 0.5:
        Teff_guess = 4000.
    else:
        Teff_guess = 10000.
    rdstar_guess = 1.e-10
        
    log10Teff_guess = np.log10(Teff_guess)
    log10rdstar_guess = np.log10(rdstar_guess)
    
    nll = lambda *args: -lnlike_star_blackbody(*args)
    
    
    result = minimize(nll, [log10Teff_guess, log10rdstar_guess], args = (cent_wavs_use, nuFnu_use, nuFnu_err_use))
    
    log10Teff_opt, log10rdstar_opt = result["x"]
    
    Teff_opt = 10.**log10Teff_opt
    rdstar_opt = 10.**log10rdstar_opt
    
    nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)

    nuFnu_star_test = nuFnu_star[:-4]
    
    initial_fit_fail = False
    
    if Teff_opt > 25000. or Teff_opt < 1000. or (rdstar_opt == 0.0) or (rdstar_opt > 0.004) or (Teff_opt > 13000. and subj.bjmag_tycho > subj.vjmag_tycho) or (Teff_opt > 7550. and subj.bjmag_tycho > (subj.vjmag_tycho + 0.35)):
        initial_fit_fail = True
        print 'Initial fit fail. Adding filter and retrying.'
        
    while filter_cut < -2 and initial_fit_fail:
        Teff_guess = 7000.
        rdstar_guess = 1.e-10
        
        log10Teff_guess = np.log10(Teff_guess)
        log10rdstar_guess = np.log10(rdstar_guess)
        
        filter_cut += 1
        print filter_cut
        
        cent_wavs_use_spec = cent_wavs_full_use[:filter_cut]
        nuFnu_use_spec = nuFnu_full_use[:filter_cut]
        nuFnu_err_use_spec = nuFnu_full_err_use[:filter_cut]
        
        
        result = minimize(nll, [log10Teff_guess, log10rdstar_guess], args = (cent_wavs_use_spec, nuFnu_use_spec, nuFnu_err_use_spec))
        
        log10Teff_opt, log10rdstar_opt = result["x"]
    
        Teff_opt = 10.**log10Teff_opt
        rdstar_opt = 10.**log10rdstar_opt
            
        if Teff_opt < 25000. or Teff_opt > 1000. or (rdstar_opt == 0.0) or (rdstar_opt < 0.004):
            print "Fit successful."
            initial_fit_fail = False
            
    if filter_cut > -3 and initial_fit_fail:
        print "Fitting unsuccessful even when including W2 and blue-ward."
        subj.fitfail = True
        return subj
                
    nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)        

    use_next = ((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) < 10.
    
    extra_check = False
    
    if Teff_opt < 5000.:
        subj.use_models = True
    elif Teff_opt < 7000. and not use_next:
        subj.use_models = True
    elif Teff_opt > 5000. and Teff_opt < 7000. and use_next:
        filter_cut += 1
        result = minimize(nll, [log10Teff_opt, log10rdstar_opt], args = (cent_wavs_full_use[:filter_cut], nuFnu_full_use[:filter_cut], nuFnu_full_err_use[:filter_cut]))
        log10Teff_opt_new, log10rdstar_opt_new = result["x"]
        if (10.**log10Teff_opt) < 7000.:
            subj.use_models = True
            extra_check = True
        
    model_star_plotting = False
    
    if subj.use_models:
        cent_wavs_fit = cent_wavs_full_use[:filter_cut]
        mags_fit = mags_full_use[:filter_cut]
        magerrs_fit = magerrs_full_use[:filter_cut]
        Teff_use, logg_use, rdstar_use = star_fitter_models(cent_wavs_fit, mags_fit, magerrs_fit, [log10Teff_opt, log10rdstar_opt])

        model_mags = btsettl_models_dict[(Teff_use, logg_use)][:-3]
        
        model_mags_at_d = model_mags - 5.*np.log10(rdstar_use)
        
        model_mags_full = btsettl_models_dict[(Teff_use, logg_use)]
        model_mags_full_at_d = model_mags_full - (5.*np.log10(rdstar_use))
        print len(model_mags_full_at_d)

        chi2model = np.sum(np.array([((mags_fit[i] - model_mags_at_d[i])**2) for i in range(len(mags_fit))]) / magerrs_fit) / len(mags_fit)
        
        fluxes_at_d = np.zeros(mags_full_use.size)
        
        for i in range(fluxes_at_d.size):
            index = cent_wavs_dict[cent_wavs_full_use[i]]
            print index, btsettl_column_labels[2+index], cent_wavs_dict_keys[index], filterzps_full_use[i]
            
            fluxes_at_d[i] = filterzps_full_use[i] * 10.**(-0.4*model_mags_full_at_d[index]) * 10.**(-23.)
            
            
        nuFnu_model = (c / cent_wavs_full_use) * fluxes_at_d
        
        nuFnu_remain = nuFnu_full_use - nuFnu_model
        
        if ((((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) and filter_cut > -4) or (((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 10.) or (filter_cut > -3):
            use_next = False
            
        if use_next:
            while use_next:
                print subj.filts_to_use[filter_cut], 'not in excess. Refitting with', subj.filts_to_use[filter_cut], 'included.'

                filter_cut += 1
            
                cent_wavs_refit = cent_wavs_full_use[:filter_cut]
                mags_refit = mags_full_use[:filter_cut]
                magerrs_refit = magerrs_full_use[:filter_cut]

                Teff_use, logg_use, rdstar_use = star_fitter_models(cent_wavs_refit, mags_refit, magerrs_refit, [log10Teff_opt, log10rdstar_opt])

                model_mags_refit = btsettl_models_dict[(Teff_use, logg_use)][:-2]
        
                model_mags_refit_at_d = model_mags_refit - 5.*np.log10(rdstar_use)
        
                model_mags_refit_full = btsettl_models_dict[(Teff_use, logg_use)]
                model_mags_refit_full_at_d = model_mags_refit_full - (5.*np.log10(rdstar_use))

                fluxes_at_d = np.zeros(mags_full_use.size)
        
                for i in range(fluxes_at_d.size):
                    index = cent_wavs_dict[cent_wavs_full_use[i]]
            
                    fluxes_at_d[i] = filterzps_full_use[i] * (10.**(-0.4*model_mags_refit_full_at_d[index])) * (10.**(-23.))
                
                nuFnu_model = (c / cent_wavs_full_use) * fluxes_at_d
        
                nuFnu_remain = nuFnu_full_use - nuFnu_model
            
                print filter_cut, nuFnu_full_use[filter_cut], nuFnu_model[filter_cut], nuFnu_full_err_use[filter_cut], ((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut])
            
                if ((((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) and filter_cut > -4) or (((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 10.) or (filter_cut > -3):
                    use_next = False

        subj.log10Teffguess = np.log10(Teff_use)
        subj.Teffguess = Teff_use
        subj.logg_guess = logg_use
        subj.log10rdstarguess = np.log10(rdstar_use)
        subj.filter_cut = filter_cut
                        
        subj.nuFnu_star = nuFnu_model
            
        teffpull = '0'+str(int(Teff_use/100))
        loggpull = str(logg_use)
        
        spect_file = '..\\..\\BTSettlstuff_use\\BT-Settl_M-0.0a+0.0\\lte'+teffpull+'.0-'+loggpull+'-0.0a+0.0.BT-Settl.spec.7'
        
        spec_X, spec_S, spec_dict = fluxdrive_plot(spect_file,1)
                
        flux_spec_S = spec_X * spec_S * (rdstar_use**2)
        
        subj.nuFnu_star_plotting_temp = np.interp(plotting_xvec_angstroms, spec_X, flux_spec_S)
        model_star_plotting = True
        
        
    else:
        if use_next:
            while use_next:
                print subj.filts_to_use[filter_cut], 'not in excess. Refitting with', subj.filts_to_use[filter_cut], 'included.'
                filter_cut += 1
                
                result = minimize(nll, [log10Teff_opt, log10rdstar_opt], args = (cent_wavs_full_use[:filter_cut], nuFnu_full_use[:filter_cut], nuFnu_full_err_use[:filter_cut]))
    
                log10Teff_opt, log10rdstar_opt = result["x"]
                
                Teff_opt = 10.**log10Teff_opt
                rdstar_opt = 10.**log10rdstar_opt

                nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)

                nuFnu_star_test = nuFnu_star[:filter_cut]
    
                nuFnu_remain = nuFnu_full_use - nuFnu_star
        
                print filter_cut, nuFnu_full_use[filter_cut], nuFnu_star[filter_cut], nuFnu_full_err_use[filter_cut], ((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut])
                if ((((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) and filter_cut > -4) or (((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 10.) or (filter_cut > -3):
                    use_next = False
            subj.log10Teffguess = np.log10(Teff_opt)
            subj.log10rdstarguess = np.log10(rdstar_opt)   
            
        else:
            subj.log10Teffguess = np.log10(Teff_opt)
            subj.log10rdstarguess = np.log10(rdstar_opt)
            subj.filter_cut = filter_cut
                
        if not model_star_plotting:
            subj.nuFnu_star = blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, (10.**subj.log10Teffguess) * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * ((10.**subj.log10rdstarguess)**2) * u.cm * u.cm * u.s / u.erg
        
        if not model_star_plotting:    
            subj.nuFnu_star_plotting_temp = blackbody_lambda(plotting_xvec_angstroms * u.AA, (10.**subj.log10Teffguess)*u.K) * np.pi * u.sr * (plotting_xvec_angstroms * u.AA) * ((10.**subj.log10rdstarguess)**2) * u.cm * u.cm * u.s / u.erg
    
     
    subj.nuFnu_disk = nuFnu_full_use - subj.nuFnu_star
    subj.nuFnu_disk_errs = nuFnu_full_err_use
                    
    test_fit = subj.nuFnu_disk[:filter_cut] / subj.nuFnu_disk_errs[:filter_cut]
     
    test_fit_matches = 0
    
    for i in range(len(test_fit)):
        if np.abs(test_fit[i]) < 5.:
            test_fit_matches += 1
    
    print test_fit_matches
    
    if test_fit_matches > np.ceil(float(len(test_fit))/2.):
        subj.good_star_fit = True
    elif subj.use_models:
        print "Model fit is bad stellar fit--will try blackbody fit."
        result_test = minimize(nll, [log10Teff_opt, log10rdstar_opt], args = (cent_wavs_full_use[:filter_cut], nuFnu_full_use[:filter_cut], nuFnu_full_err_use[:filter_cut]))
        
        log10Teff_opt, log10rdstar_opt = result["x"]
                
        Teff_opt = 10.**log10Teff_opt
        rdstar_opt = 10.**log10rdstar_opt

        nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)

        nuFnu_star_test = nuFnu_star[:filter_cut]
    
        nuFnu_remain = nuFnu_full_use - nuFnu_star
        
        test_fit_bb = nuFnu_full_use[:filter_cut] / nuFnu_full_err_use[:filter_cut]
        
        test_fit_matches_bb = 0
        
        for i in range(len(test_fit_bb)):
            if np.abs(test_fit_bb[i]) < 5.:
                test_fit_matches_bb += 1
                
        if test_fit_matches_bb > test_fit_matches:
            subj.log10Teffguess = log10Teff_opt
            subj.log10rdstarguess = log10rdstar_opt
            subj.nuFnu_star = blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, (10.**subj.log10Teffguess) * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * ((10.**subj.log10rdstarguess)**2) * u.cm * u.cm * u.s / u.erg
            subj.nuFnu_star_plotting_temp = blackbody_lambda(plotting_xvec_angstroms * u.AA, (10.**subj.log10Teffguess)*u.K) * np.pi * u.sr * (plotting_xvec_angstroms * u.AA) * ((10.**subj.log10rdstarguess)**2) * u.cm * u.cm * u.s / u.erg
        else:
            print 'Bad stellar fit when using blackbody as well--will not fit disk.'
    else:
        print "Bad stellar fit--will not fit disk."
            
    
    subj.sig_disk = subj.nuFnu_disk[-4:] / subj.nuFnu_disk_errs[-4:]
    
    subj.num_excesses = subj.sig_disk[subj.sig_disk > 5].size
    
    if subj.num_excesses < 1:
        subj.num_excesses = subj.sig_disk[subj.sig_disk > 3].size
    
    return subj
	