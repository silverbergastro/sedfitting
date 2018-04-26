import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.constants import h,k,c
from scipy.optimize import minimize
from astropy import units as u
from astropy.analytic_functions import blackbody_lambda, blackbody_nu
import time
import emcee
#import sedfitter

import astropy
print astropy.__version__

import scipy
print scipy.__version__

import matplotlib
print matplotlib.__version__

print np.__version__

#import winsound

print "Dependencies imported"

filternamevec = ['PSg','PSr','PSi','PSz','PSy','Gaia','BJohnson','VJohnson','SDSSg','SDSSr','SDSSi','J','H','K','W1','W2','W3','W4']
filterzps = [3631., 3631., 3631., 3631., 3631., 2861.3, 4000.87, 3597.28, 3631., 3631., 3631., 1594., 1024., 666.7, 309.54, 171.787, 31.674, 8.363]
filtercentwav = [0.4810, 0.6170, 0.7520, 0.8660, 0.9620, 0.673, 0.4361, 0.5448, 0.4770, 0.6231, 0.7625, 1.235, 1.662, 2.159, 3.35, 4.60, 11.56, 22.08]

filterzps_short = filterzps[-7:]
#filterzps_nops = filterzps[5:]
#filterzps_nogaia = filterzps[6:]

filterzps_dict = {filtercentwav[i]: filterzps[i] for i in range(len(filtercentwav))}

class FullSubject:
    def __init__(self, vec):
        self.zooniverse_id = vec[8]
        self.wiseid = vec[7]
        self.ra = float(vec[1])
        self.dec = float(vec[2])
        self.glong = float(vec[9])
        self.glat = float(vec[10])

        self.jmag = float(vec[19])
        self.jmagerr = float(vec[20])
        self.hmag = float(vec[21])
        self.hmagerr = float(vec[22])
        self.kmag = float(vec[23])
        self.kmagerr = float(vec[24])
        tempw1mag = float(vec[11])
        tempw1magerr = float(vec[12])
        tempw2mag = float(vec[13])
        tempw2magerr = float(vec[14])
        self.w3mag = float(vec[15])
        self.w3magerr = float(vec[16])
        self.w4mag = float(vec[17])
        self.w4magerr = float(vec[18])
        
        if tempw1mag > 8.:
            self.w1mag = tempw1mag + 0.
            self.w1magerr = tempw1magerr + 0.
        else:
            self.w1mag = tempw1mag + (0. - 0.1359 + (0.0396*tempw1mag) - (0.0023*tempw1mag*tempw1mag)) - 0.031
            self.w1magerr = (((1. + 0.0396 - (0.0046*tempw1mag))**2) * (tempw1magerr**2))**0.5
            
        if tempw2mag > 6.7:
            self.w2mag = tempw2mag + 0.
            self.w2magerr = tempw2magerr + 0.
        elif tempw2mag > 5.4:
            self.w2mag = tempw2mag + (-0.3530 + (0.8826*tempw2mag) - (0.2380*(tempw2mag**2)) + (0.0170*(tempw2mag**3))) + 0.004
            self.w2magerr = (((1. + 0.8826 - (0.4760 * tempw2mag) + (0.0510 * (tempw2mag**2)))**2) * (tempw2magerr**2))**0.5
        else:
            self.w2mag = tempw2mag + (1.5777 - (0.3495*tempw2mag) + (0.0160*(tempw2mag**2))) + 0.004
            self.w2magerr = (((1. - 0.3495 + (2.*0.0160*tempw2mag))**2) * (tempw2magerr**2))**0.5

        self.psgmag = float(vec[54])
        self.psgmagerr = float(vec[55])
        self.psimag = float(vec[56])
        self.psimagerr = float(vec[57])
        self.psrmag = float(vec[58])
        self.psrmagerr = float(vec[59])
        self.psymag = float(vec[60])
        self.psymagerr = float(vec[61])
        self.pszmag = float(vec[62])
        self.pszmagerr = float(vec[63])
        
        self.has_ps_gminr = (not math.isnan(self.psgmag)) and (not math.isnan(self.psrmag))
        if self.has_ps_gminr:
            self.psgminr = self.psgmag - self.psrmag
            self.psgminrerr = ((self.psgmagerr**2) + (self.psrmagerr**2))**0.5
        else:
            self.psgminr = float('NaN')
        
        self.sourcevec = vec
        
        self.gaiamag = float(vec[43])
        self.gaiamagerr = float(vec[45])
        #if not math.isnan(float(vec[80])):
        #    self.bjmag = float(vec[80])
        #    self.bjmagerr = float(vec[81])
        #elif not math.isnan(float(vec[35])) and (float(vec[35]) > 0.) and (float(vec[35]) < 5.) and (float(vec[29]) < 50.):
        #    self.bjmag = float(vec[29])
        #    self.bjmagerr = float(vec[35])
        self.bjmag_tycho = float(vec[80])
        self.bjmagerr_tycho = float(vec[81])        
        self.vjmag_tycho = float(vec[82])
        self.vjmagerr_tycho = float(vec[83])
        
        self.has_tycho_vmag = (not math.isnan(self.vjmag_tycho))
        self.has_gaia = (not math.isnan(self.gaiamag))
        self.has_tycho_bmag = (not math.isnan(self.bjmag_tycho))
        
        self.apass_vmag = float(vec[27])
        self.apass_bmag = float(vec[29])
        self.apass_gmag = float(vec[30])
        self.apass_rmag = float(vec[31])
        self.apass_imag = float(vec[32])
        
        self.apass_vmagerr = float(vec[33])
        self.apass_bmagerr = float(vec[35])
        self.apass_gmagerr = float(vec[36])
        self.apass_rmagerr = float(vec[37])
        self.apass_imagerr = float(vec[38])

        self.umag = float(vec[84])
        self.gmag = float(vec[86])
        self.rmag = float(vec[88])
        self.imag = float(vec[90])
        self.zmag = float(vec[92])
        
        self.umagerr = float(vec[85])
        self.gmagerr = float(vec[87])
        self.rmagerr = float(vec[89])
        self.imagerr = float(vec[91])
        self.zmagerr = float(vec[93])
        
        self.use_tycho = vec[94] in ['TRUE','True','true']
        self.use_gaia = vec[95] in ['TRUE','True','true']
        self.use_ps = vec[96] in ['TRUE','True','true']
        self.use_sdss = vec[97] in ['TRUE','True','true']

        
        filtinputvec = [self.psgmag, self.psrmag, self.psimag, self.pszmag, self.psymag, self.gaiamag, self.bjmag_tycho, self.vjmag_tycho, self.gmag, self.rmag, self.imag, self.jmag, self.hmag, self.kmag, self.w1mag, self.w2mag, self.w3mag, self.w4mag]
        filterrinputvec = [self.psgmagerr, self.psrmagerr, self.psimagerr, self.pszmagerr, self.psymagerr, self.gaiamagerr, self.bjmagerr_tycho, self.vjmagerr_tycho, self.gmagerr, self.rmagerr, self.imagerr, self.jmagerr, self.hmagerr, self.kmagerr, self.w1magerr, self.w2magerr, self.w3magerr, self.w4magerr]
        
        self.filts_temp = []
        self.mags_temp = []
        self.magerrs_temp = []
        self.fluxes_temp = []
        self.fluxerrs_temp = []
        self.centwavs_temp = []
        self.filterzps_temp = []
        
        self.filts_nops_temp = []
        self.mags_nops_temp = []
        self.magerrs_nops_temp = []
        self.fluxes_nops_temp = []
        self.fluxerrs_nops_temp = []
        self.centwavs_nops_temp = []
        self.filterzps_nops_temp = []
        
        self.filts_to_use_temp = []
        self.mags_to_use_temp = []
        self.magerrs_to_use_temp = []
        self.fluxes_to_use_temp = []
        self.fluxerrs_to_use_temp = []
        self.centwavs_to_use_temp = []
        self.filterzps_to_use_temp = []
        
        for i in range(len(filtinputvec)):
            #print filtinputvec[i]
            if math.isnan(filtinputvec[i]):
                continue
            else:
                self.filts_temp.append(filternamevec[i])
                self.mags_temp.append(filtinputvec[i])
                self.magerrs_temp.append(filterrinputvec[i])
                self.fluxes_temp.append(filterzps[i] * (10.**(-0.4*filtinputvec[i])))
                self.fluxerrs_temp.append(0.4*np.log(10.)*filterrinputvec[i]*(filterzps[i] * (10.**(-0.4*filtinputvec[i]))))
                self.centwavs_temp.append(filtercentwav[i])
                self.filterzps_temp.append(filterzps[i])

        for i in range(len(self.filts_temp)):
            if 'PS' not in self.filts_temp[i]:
                #if (self.has_tycho_vmag or self.has_tycho_bmag) and ('Gaia' not in self.filts_temp[i]):
                #    self.filts_nogaia_temp.append(self.filts_temp[i])
                #    self.mags_nogaia_temp.append(self.mags_temp[i])
                #    self.magerrs_nogaia_temp.append(self.magerrs_temp[i])
                #    self.fluxes_nogaia_temp.append(self.fluxes_temp[i])
                #    self.fluxerrs_nogaia_temp.append(self.fluxerrs_temp[i])
                #    self.centwavs_nogaia_temp.append(self.centwavs_temp[i])
                #    self.filterzps_nogaia_temp.append(self.filterzps_temp[i])

                self.filts_nops_temp.append(self.filts_temp[i])
                self.mags_nops_temp.append(self.mags_temp[i])
                self.magerrs_nops_temp.append(self.magerrs_temp[i])
                self.fluxes_nops_temp.append(self.fluxes_temp[i])
                self.fluxerrs_nops_temp.append(self.fluxerrs_temp[i])
                self.centwavs_nops_temp.append(self.centwavs_temp[i])
                self.filterzps_nops_temp.append(self.filterzps_temp[i])
                
        for i in range(len(self.filts_temp)):
            if 'PS' in self.filts_temp[i] and not self.use_ps:
                continue
            elif 'SDSS' in self.filts_temp[i] and not self.use_sdss:
                continue
            elif 'Johnson' in self.filts_temp[i] and not self.use_tycho:
                continue
            elif 'Gaia' in self.filts_temp[i] and not self.use_gaia:
                continue
            else:
                self.filts_to_use_temp.append(self.filts_temp[i])
                self.mags_to_use_temp.append(self.mags_temp[i])
                self.magerrs_to_use_temp.append(self.magerrs_temp[i])
                self.fluxes_to_use_temp.append(self.fluxes_temp[i])
                self.fluxerrs_to_use_temp.append(self.fluxerrs_temp[i])
                self.centwavs_to_use_temp.append(self.centwavs_temp[i])
                self.filterzps_to_use_temp.append(self.filterzps_temp[i])
                #print self.filts_to_use_temp
                
                
                
        self.filts_long = np.array(self.filts_temp)
        self.mags_long = np.array(self.mags_temp)
        self.magerrs_long = np.array(self.magerrs_temp)
        self.fluxes_long = np.array(self.fluxes_temp) * 1.e-23
        self.fluxerrs_long = np.array(self.fluxerrs_temp) * 1.e-23
        self.centwavs_microns_long = np.array(self.centwavs_temp)
        self.centwavs_meters_long = self.centwavs_microns_long * 1.e-6
        self.centwavs_Hz_long = c/self.centwavs_meters_long
        self.nuFnu_long = self.centwavs_Hz_long * self.fluxes_long
        self.nuFnuerrs_long = self.centwavs_Hz_long * self.fluxerrs_long
        
        self.filts_nops = np.array(self.filts_nops_temp)
        self.mags_nops = np.array(self.mags_nops_temp)
        self.magerrs_nops = np.array(self.magerrs_nops_temp)
        self.fluxes_nops = np.array(self.fluxes_nops_temp) * 1.e-23
        self.fluxerrs_nops = np.array(self.fluxerrs_nops_temp) * 1.e-23
        self.centwavs_microns_nops = np.array(self.centwavs_nops_temp)
        self.centwavs_meters_nops = self.centwavs_microns_nops * 1.e-6
        self.centwavs_Hz_nops = c/self.centwavs_meters_nops
        self.nuFnu_nops = self.centwavs_Hz_nops * self.fluxes_nops
        self.nuFnuerrs_nops = self.centwavs_Hz_nops * self.fluxerrs_nops
        self.filterzps_nops = np.array(self.filterzps_nops_temp)
        
        self.filts_to_use = np.array(self.filts_to_use_temp)
        self.mags_to_use = np.array(self.mags_to_use_temp)
        self.magerrs_to_use = np.array(self.magerrs_to_use_temp)
        self.fluxes_to_use = np.array(self.fluxes_to_use_temp) * 1.e-23
        self.fluxerrs_to_use = np.array(self.fluxerrs_to_use_temp) * 1.e-23
        self.centwavs_microns_to_use = np.array(self.centwavs_to_use_temp)
        self.centwavs_meters_to_use = self.centwavs_microns_to_use * 1.e-6
        self.centwavs_Hz_to_use = c/self.centwavs_meters_to_use
        self.nuFnu_to_use = self.centwavs_Hz_to_use * self.fluxes_to_use
        self.nuFnuerrs_to_use = self.centwavs_Hz_to_use * self.fluxerrs_to_use
        self.filterzps_to_use = np.array(self.filterzps_to_use_temp)
        
        self.centwavs_microns = self.centwavs_microns_long[-7:]
        self.centwavs_meters = self.centwavs_microns * 1.e-6
        self.centwavs_Hz = c/self.centwavs_meters
        self.mags = self.mags_long[-7:]
        self.magerrs = self.magerrs_long[-7:]
        self.nuFnu = self.nuFnu_long[-7:]
        self.nuFnuerrs = self.nuFnuerrs_long[-7:]

        #self.centwavs_microns_optical = self.centwavs_microns_long[5:]
        #self.centwavs_meters_optical = self.centwavs_microns_optical * 1.e-6
            
        
        self.Teff = None
        self.Teff_err_low = None
        self.Teff_err_high = None
        self.logg = None
        self.logg_err_low = None
        self.logg_err_high = None
        self.rdstar = None
        self.rdstar_err_low = None
        self.rdstar_err_high = None
        self.Tdisk = None
        self.Tdisk_err_low = None
        self.Tdisk_err_high = None
        self.xdisk = None
        self.xdisk_err_low = None
        self.xdisk_err_high = None
        self.fir = None
        self.fir_err_low = None
        self.fir_err_high = None
        
        self.nuFnu_star = None
        self.nuFnu_disk = None
        
        self.nuFnu_star_plotting_temp = None
        self.nuFnu_disk_plotting_temp = None
        self.nuFnu_model_plotting_temp = None

        self.nuFnu_disk_plotting_temp_powerlaw = None
        self.nuFnu_model_plotting_temp_powerlaw = None
        self.nuFnu_disk_plotting_temp_blackbody = None
        self.nuFnu_model_plotting_temp_blackbody = None
        
        self.nuFnu_star_plotting = None
        self.nuFnu_disk_plotting = None
        self.nuFnu_model_plotting = None        
        
        self.sig_disk = None
        self.num_excesses = None
        
        self.good_star_fit = False
        self.good_disk_fit = False
        
        self.chistar = None
        self.chidisk = None
        
        self.fitfail = False
        
        self.use_models = False
        
        self.log10Teffguess = None
        self.log10rdstarguess = None
        self.Teffguess = None
        self.logg_guess = None
        self.log10Tdiskguess = None
        self.log10xdiskguess = None
        
        self.filter_cut = None
        self.sampler = None
		
		self.alpha_guess = None
		self.log10beta_guess = None
        
        self.powerlaw = False
        self.alpha = None
		self.alpha_err_high = None
		self.alpha_err_low = None
        self.log10beta = None
        self.beta = None
		self.beta_err_high = None
		self.beta_err_low = None

    def __str__(self):
        s = ''
        
		string_vec = [self.zooniverse_id, self.wiseid, str(self.Teff), str(self.Teff_err_low), str(self.Teff_err_high), str(self.logg), str(self.logg_err_low), str(self.logg_err_high), str(self.rdstar), str(self.rdstar_err_low), str(self.rdstar_err_high), str(self.Tdisk), str(self.Tdisk_err_low), str(self.Tdisk_err_high), str(self.xdisk), str(self.xdisk_err_low), str(self.xdisk_err_high), str(self.fir), str(self.fir_err_low), str(self.fir_err_high), str(self.alpha), str(self.alpha_err_low), str(self.alpha_err_high), str(self.beta), str(self.beta_err_low), str(self.beta_err_high)]

		for entry in string_vec:
		    s = s + entry + ','
			
		s = s[:-1]
		
        return s
    
print "Class read in"

def get_data(filename):
    df = pd.read_csv(filename, low_memory = False)
    
    data = df.values
    
    return data
	
inputdata = get_data('sed_input_data.csv')

#print inputdata[1,:]

columnlabels = inputdata[1,:]

columncount = columnlabels.size

columnlabel_index_dict = {}

for i in range(columncount):
    columnlabel_index_dict[columnlabels[i]] = i

    
#from pprint import pprint
#pprint(columnlabel_index_dict)
    
inputdata_use = inputdata[5:,:]

#print inputdata_use.shape

num_subjs = inputdata_use[:,0].size

raw_subjs = []

for i in range(num_subjs):
    raw_subjs.append(FullSubject(inputdata_use[i,:]))
    
print "Subjects read in"

#has_tycho_subjs = [i for i in range(num_subjs) if raw_subjs[i].has_tycho]

#print has_tycho_subjs

#print columnlabels
#print raw_subjs[180].sourcevec

#print raw_subjs[180].sourcevec[columnlabel_index_dict[' P_gMeanPSFMag']]

#print raw_subjs[180].filts
#print raw_subjs[180].mags
#print raw_subjs[180].magerrs
#print raw_subjs[180].fluxes
#print raw_subjs[180].fluxerrs
#print raw_subjs[180].centwavs_microns
#print raw_subjs[0].zooniverse_id
#print raw_subjs[0].wiseid
#print raw_subjs[40].has_tycho

#if raw_subjs[40].has_tycho:
#    plt.errorbar(raw_subjs[40].hastyc_mag_fitting_centwavs_microns, raw_subjs[40].hastyc_mag_fitting_nuFnu, yerr=raw_subjs[40].hastyc_mag_fitting_nuFnu_errs, fmt='b.')
#else:
#plt.errorbar(raw_subjs[0].centwavs_microns, raw_subjs[0].nuFnu, yerr=raw_subjs[0].nuFnuerrs, fmt='b.')
    
#min_nuFnu_logs = min(np.log10(raw_subjs[0].nuFnu))
#max_nuFnu_logs = max(np.log10(raw_subjs[0].nuFnu))

#ymin = float(np.floor((min_nuFnu_logs-0.5)*2.))/2.
#ymax = float(np.ceil(2.*(max_nuFnu_logs+0.5)))/2.

#print ymin, ymax

#plt.xlabel(r'$\mathrm{Wavelength(\mu m)}$', fontsize=20)
#plt.ylabel(r'$\mathrm{\nu F_{\nu} (erg\,s^{-1}\,cm^{-2})}$', fontsize=20)


#plt.xscale("log", nonposx='clip')
#plt.yscale("log", nonposy='clip')
#plt.xlim([0.25, 50.])
#plt.ylim([10.**ymin, 10.**ymax])
#plt.show()

#print raw_subjs[2].zooniverse_id
#print raw_subjs[2].wiseid
#print raw_subjs[30].has_tycho

#if raw_subjs[30].has_tycho:
#    plt.errorbar(raw_subjs[30].hastyc_mag_fitting_centwavs_microns, raw_subjs[30].hastyc_mag_fitting_nuFnu, yerr=raw_subjs[30].hastyc_mag_fitting_nuFnu_errs, fmt='b.')
#else:
#plt.errorbar(raw_subjs[2].centwavs_microns_nops, raw_subjs[2].nuFnu_nops, yerr=raw_subjs[2].nuFnuerrs_nops, fmt='b.')

#min_nuFnu_logs = min(np.log10(raw_subjs[2].nuFnu_nops))
#max_nuFnu_logs = max(np.log10(raw_subjs[2].nuFnu_nops))

#ymin = float(np.floor((min_nuFnu_logs-0.5)*2.))/2.
#ymax = float(np.ceil(2.*(max_nuFnu_logs+0.5)))/2.

#print ymin, ymax

#plt.xlabel(r'$\mathrm{Wavelength(\mu m)}$', fontsize=20)
#plt.ylabel(r'$\mathrm{\nu F_{\nu} (erg\,s^{-1}\,cm^{-2})}$', fontsize=20)


#plt.xscale("log", nonposx='clip')
#plt.yscale("log", nonposy='clip')
#plt.xlim([0.25, 50.])
#plt.ylim([10.**ymin, 10.**ymax])
#plt.show()

def fluxdrive_plot(spectname, binsize):
    spec_X, spec_S, spec_dict = import_spectrum(spectname, binsize)
    return spec_X, spec_S, spec_dict

def import_spectrum(spectname, binsize):
    spect_Xlist = []
    spect_Slist = []
    spect_dict = {}
    
    with open(spectname) as f:
        spectfilelist = f.readlines()
        
    testline = spectfilelist[0]
    test_res = [pos for pos, char in enumerate(testline) if char == '-']
    line_start = testline.index('1')
    
    flag1 = False
    flag2 = False
    
    flag1 = (test_res[0] < (7 + line_start))
    flag2 = (test_res[1] > (20 + line_start))
    
    #print flag1, flag2
    
    for line in spectfilelist:
        if (flag1 and flag2):
            line_use = line[:13] + ' ' + line[13:25] + ' ' + line[25:]
        elif flag1:
            line_use = line[:13] + ' ' + line[13:25] + ' ' + line[25:]
        elif flag2:
            line_use = line[:25] + ' ' + line[25:]
        else:
            line_use = line
            
        datavec = line_use.split()
        xstr = datavec[0]
        sstr = datavec[1]
        sstr1 = sstr.replace('D','e')
        #print datavec
        #print sstr1
        
        x = float(xstr)
        s = float(sstr1)
        
        spect_Xlist.append(x)
        spect_Slist.append((10.**(s-8.)))
        spect_dict[x] = s
        
    spect_X_binned = []
    spect_S_binned = []
    
    #ents_per_bin = binsize*20.
    
    spect_X = np.array(spect_Xlist)
    spect_S = np.array(spect_Slist)
    
    return spect_X, spect_S, spect_dict

def lnlike_star_blackbody(theta, wav, flux, fluxerr):
    log10Teff, log10rdstar = theta

    Teff = 10.**log10Teff
    rdstar = 10.**log10rdstar
    
    centwavs_angstroms = wav * 1.e10 * u.AA
    centwavs_meters = wav * u.m
    
    flux_lam_temp = blackbody_lambda(centwavs_angstroms, Teff*u.K) * np.pi * u.sr
    
    flux_lam = flux_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg

    model = flux_lam * (rdstar**2)
    
    #print Teff, rdstar
    #print "flux:", flux
    #print "model:", model

    inv_sigma2 = 1./(fluxerr**2)
    
    #print -0.5*np.sum((((flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))
    
    return -0.5*np.sum((((flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))
    
def star_fitter_blackbody(centwavs, log10Teff, log10rdstar):
    centwavs_angstroms = centwavs * 1.e10 * u.AA
    centwavs_meters = centwavs * u.m
    
    #print log10Teff, log10rdstar
    
    Teff = 10.**(log10Teff)
    rdstar = 10.**(log10rdstar)
    
    #print Teff, rdstar
    
    #print centwavs_angstroms
    #print Teff
    
    #print blackbody_lambda(centwavs_angstroms[0], Teff*u.K)
    
    #flux_lam_temp = []
    #for wav in centwavs_angstroms:
    #    flux_lam_temp.append(blackbody_lambda(wav, Teff*u.K) * np.pi * u.sr)
    
    flux_lam_temp = blackbody_lambda(centwavs_angstroms, Teff*u.K) * np.pi * u.sr 
    
    #print flux_lam_temp
    
    flux_lam = flux_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg
    
    #print flux_lam 
    
    return flux_lam * (rdstar**2)


def get_chi2(mags, magerrs, result_mags):
    chis = np.zeros(mags.size)
    
    for i in range(mags.size):
        chis[i] = ((mags[i] - result_mags[i])**2)/magerrs[i]
        
    sumchis = np.sum(chis)
    
    return sumchis

cent_wavs_dict = {}
    
cent_wavs_dict_keys = np.array([0.673, 0.4361, 0.5448, 0.3543, 0.4770, 0.6231, 0.7625, 0.9134, 1.235, 1.662, 2.159, 3.35, 4.60, 11.56, 22.08]) * 1.e-6

for i in range(cent_wavs_dict_keys.size):
    cent_wavs_dict[cent_wavs_dict_keys[i]] = i
    
btsettl_column_labels = ['Teff', 'Logg','G','B','V','u','g','r','i','z','J','H','K','W1_W10','W2_W10','W3_W10','W4_W10']

    
#print cent_wavs_dict

def star_fitter_models(centwavs, mags, magerrs, guess):
    fit_results_dict = {model: None for model in btsettl_models_dict.keys()}
    fit_chi_dict = {model: None for model in btsettl_models_dict.keys()}
    
    magerrs_use = np.array(magerrs)
    
    for i in range(centwavs.size):
        if centwavs[i] == cent_wavs_dict_keys[0]:
            magerrs_use[i] = magerrs[i] * 10.
    
    log10Teff_guess = guess[0]
    Teff_guess = 10.**log10Teff_guess
    difTeffguess = 7000. - Teff_guess
    
    mindifTeffguess = Teff_guess - difTeffguess
    
    log10rdstar_guess = guess[1]
    
    keys_use = []
    
    for model in btsettl_models_dict.keys():
        Teff = model[0]
        logg = model[1]
        
        if Teff < mindifTeffguess:
            continue
        else:
            keys_use.append(model)
            
            def lnlike_rdstar(theta, centwavs, mags, magerrs):
                log10rdstar = theta[0]
                
                magerrs_use = np.array(magerrs)
                
                fitmagsfull = btsettl_models_dict[(Teff, logg)]
                
                fitmagsuse = []
                
                #print centwavs
                
                for i in range(len(centwavs)):
                    #print cent_wavs_dict[centwavs[i]]
                    fitmagsuse.append(fitmagsfull[cent_wavs_dict[centwavs[i]]])
                    if centwavs[i] == cent_wavs_dict_keys[0]:
                        magerrs_use[i] = 10.*magerrs[i]

                #print fitmagsuse
                
                result_mags = fitmagsuse - 5.*log10rdstar
                
                inv_sigma2 = 1./(magerrs_use**2)
                
                return -0.5*np.sum((((mags - result_mags)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))
    
            nll1 = lambda *args: -lnlike_rdstar(*args)
    
            result1 = minimize(nll1, [log10rdstar_guess], args = (centwavs, mags, magerrs))
        
            log10rdstar_opt = result1["x"][0]
            
            #popt, pcov = curve_fit(fit_rdstar, centwavs, mags, sigma=magerrs)
        
            fit_mags_full = btsettl_models_dict[(Teff, logg)]
            
            #print Teff, logg, log10rdstar_opt
            #print fit_mags_full
            
            fit_mags_use = []
            
            for i in range(len(centwavs)):
                fit_mags_use.append(fit_mags_full[cent_wavs_dict[centwavs[i]]])
                
            fit_result_mags = fit_mags_use - 5.*log10rdstar_opt
            
            #print Teff, logg, log10rdstar_opt
            #print fit_result_mags
            #print mags
            
            fit_results_dict[model] = 10.**log10rdstar_opt
            fit_chi_dict[model] = get_chi2(mags, magerrs_use, fit_result_mags)
            
            #print fit_chi_dict[model]
    
    fit_chi_dict_use = {x: fit_chi_dict[x] for x in keys_use}
    fit_results_dict_use = {x: fit_results_dict[x] for x in keys_use}
    
    minchi = min(fit_chi_dict_use.values())
    
    best_model = fit_chi_dict_use.keys()[fit_chi_dict_use.values().index(minchi)]
    
    best_rdstar = (fit_results_dict_use[best_model])
    
    return best_model[0], best_model[1], best_rdstar

plotting_logx_vec = np.linspace(np.log10(0.25), np.log10(100.), 1001)
plotting_x_vec = np.zeros(1001)
    
for i in range(1001):
    plotting_x_vec[i] = 10.**(plotting_logx_vec[i])
    
plotting_xvec_angstroms = plotting_x_vec * 1.e4

btsettl_df = pd.read_csv('btsettl_combo_readin.csv',low_memory=False)
btsettl_data = btsettl_df.values

num_models = btsettl_data[:,0].size

btsettl_models_dict = {}

#print num_models

keylist = []

for i in range(num_models):
    if float(btsettl_data[i,0]) > 2400.:
    #print btsettl_data[i,0]
    #print btsettl_data[i,1]
    #print btsettl_data[i,2:]
        key = (float(btsettl_data[i,0]), float(btsettl_data[i,1]))
        keylist.append(key)
        btsettl_models_dict[key] = btsettl_data[i,2:]

#for key in keylist:
#    btsettl_models_dict[key] = btsettl_data[i,2:]
    
def get_star_fit(subj):
    #print subj.nuFnu.size
    #print subj.nuFnuerrs.size
    print subj.filts_to_use
    nuFnu_full_use = subj.nuFnu_to_use
    nuFnu_full_err_use = subj.nuFnuerrs_to_use
    mags_full_use = subj.mags_to_use
    magerrs_full_use = subj.magerrs_to_use
    cent_wavs_full_use = subj.centwavs_meters_to_use
    filterzps_full_use = subj.filterzps_to_use
        
        
        #nuFnu_use = subj.nuFnu_nogaia[:-4]
        #nuFnu_err_use = subj.nuFnuerrs_nogaia[:-4]
        #mags_use = subj.mags_nogaia[:-4]
        #magerrs_use = subj.magerrs_nogaia[:-4]
        #cent_wavs_use = subj.centwavs_meters_nogaia[:-4]
        #filterzps_use = subj.filterzps_nogaia[:-4]
    #elif (subj.jmag < 14.5) and ((subj.jmag - subj.hmag) > 0.):
    #    nuFnu_full_use = subj.nuFnu_long
    #    nuFnu_full_err_use = subj.nuFnuerrs_long
    #    mags_full_use = subj.mags_long
    #    magerrs_full_use = subj.magerrs_long
    #    cent_wavs_full_use = subj.centwavs_meters_long
    #    
    #    nuFnu_use = subj.nuFnu_nogaia[:-4]
    #    nuFnu_err_use = subj.nuFnuerrs_nogaia[:-4]
    #    mags_use = subj.mags_nogaia[:-4]
    #    magerrs_use = subj.magerrs_nogaia[:-4]
    #    cent_wavs_use = subj.centwavs_meters_nogaia[:-4]                
    #else:
    #    nuFnu_full_use = subj.nuFnu_nops
    #    nuFnu_full_err_use = subj.nuFnuerrs_nops
    #    mags_full_use = subj.mags_nops
    #    magerrs_full_use = subj.magerrs_nops
    #    cent_wavs_full_use = subj.centwavs_meters_nops
    #    filterzps_full_use = subj.filterzps_nops
        
        #nuFnu_use = subj.nuFnu_nops[:-3]
        #nuFnu_err_use = subj.nuFnuerrs_nops[:-3]
        #mags_use = subj.mags_nops[:-3]
        #magerrs_use = subj.mags_nops[:-3]
        #cent_wavs_use = subj.centwavs_meters_nops[:-3]
        #filterzps_use = subj.filterzps_nops[:-3]
    
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

    
    #print nuFnu_full_use
    #print nuFnu_full_err_use
    #print mags_full_use
    #print magerrs_full_use
    
    #print nuFnu_full_use/nuFnu_full_err_use
    
    #mags_use = subj.mags[:-3]
    #magerrs_use = subj.magerrs[:-3]
    
    #print nuFnu_use
    #print nuFnu_err_use
    #print cent_wavs_use
    
    if mags_full_use[-7] - mags_full_use[-6] > 0.5:
        Teff_guess = 4000.
    else:
        Teff_guess = 10000.
    rdstar_guess = 1.e-10
    
    #use_w1 = False
    #use_w2 = False
    #k_excess = False
    
    log10Teff_guess = np.log10(Teff_guess)
    log10rdstar_guess = np.log10(rdstar_guess)
    
    #print nuFnu_use
    
    #popt, pcov = curve_fit(star_fitter_blackbody, cent_wavs_use, nuFnu_use, sigma = nuFnu_err_use, p0 = [logTeff_guess, logrdstar_guess])
    #Teff_opt = 10.**(popt[0])
    #rdstar_opt = 10.**(popt[1])

    nll = lambda *args: -lnlike_star_blackbody(*args)
    
    #print len(cent_wavs_use)
    #print len(nuFnu_use)
    #print len(nuFnu_err_use)
    
    result = minimize(nll, [log10Teff_guess, log10rdstar_guess], args = (cent_wavs_use, nuFnu_use, nuFnu_err_use))
    
    log10Teff_opt, log10rdstar_opt = result["x"]
    
    Teff_opt = 10.**log10Teff_opt
    rdstar_opt = 10.**log10rdstar_opt
    
    nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)

    nuFnu_star_test = nuFnu_star[:-4]
    
    #print len(nuFnu_use)
    #print len(nuFnu_star_test)
    #print len(nuFnu_err_use)
    
    #chi2test = np.sum(np.square(nuFnu_use - nuFnu_star_test)/nuFnu_err_use) / len(nuFnu_use)

    #print 'Fit W1 flux:', nuFnu_star[-4]
    #print 'Observed W1 flux:', nuFnu_full_use[-4]
    #print 'Observed W1 uncertainty:', nuFnu_full_err_use[-4]
    #print 'Difference (Observed - Fit):', nuFnu_full_use[-4] - nuFnu_star[-4]
    #print 'Significance:', (nuFnu_full_use[-4] - nuFnu_star[-4])/nuFnu_full_err_use[-4]
    #print ' '
    
    #print 'Fit K flux:', nuFnu_star[-5]
    #print 'Observed K flux:', nuFnu_full_use[-5]
    #print 'Observed K uncertainty:', nuFnu_full_err_use[-5]
    #print 'Difference (Observed - Fit):', nuFnu_full_use[-5] - nuFnu_star[-5]
    #print 'Significance:', (nuFnu_full_use[-5] - nuFnu_star[-5])/nuFnu_full_err_use[-5]
    
    #if (((nuFnu_full_use[-5] - nuFnu_star[-5])/nuFnu_full_err_use[-5]) > 5.) and (len(nuFnu_full_use[:-5]) > 2):
    #    k_excess = True
    
    #nuFnu_remain = nuFnu_full_use - nuFnu_star
    
    #if ((nuFnu_full_use[-4] - nuFnu_star[-4])/nuFnu_full_err_use[-4]) < 5.:
    #    use_w1 = True
        
    #print k_excess
    
    #if nuFnu_remain[-4] < (nuFnu_full_use[-4] - (3.*(nuFnu_full_err_use[-4]))):
    #    use_w1=True

    initial_fit_fail = False
    
    if Teff_opt > 25000. or Teff_opt < 1000. or (rdstar_opt == 0.0) or (rdstar_opt > 0.004) or (Teff_opt > 13000. and subj.bjmag_tycho > subj.vjmag_tycho) or (Teff_opt > 7550. and subj.bjmag_tycho > (subj.vjmag_tycho + 0.35)):
        initial_fit_fail = True
        print 'Initial fit fail. Adding filter and retrying.'
        #k_excess = False
        #use_w1 = True
        
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
    
        #print Teff_opt, rdstar_opt
        
        if Teff_opt < 25000. or Teff_opt > 1000. or (rdstar_opt == 0.0) or (rdstar_opt < 0.004):
            print "Fit successful."
            initial_fit_fail = False
            
    if filter_cut > -3 and initial_fit_fail:
        print "Fitting unsuccessful even when including W2 and blue-ward."
        subj.fitfail = True
        return subj
            

    #if k_excess:
    #    print 'K excess detected. Trying fitting without K.'
    #    nuFnu_full_use_short = nuFnu_full_use[:-1]
    #    nuFnu_full_err_use_short = nuFnu_full_err_use[:-1] 
    #    mags_full_use_short = mags_full_use[:-1]
    #    magerrs_full_use_short = magerrs_full_use[:-1]
    #    cent_wavs_full_use_short = cent_wavs_full_use[:-1]

    #    log10Teff_guess = np.log10(10000.)
    #    log10rdstar_guess = np.log10(1.e-10)

    #    result = minimize(nll, [log10Teff_guess, log10rdstar_guess], args = (cent_wavs_use, nuFnu_use, nuFnu_err_use))
    
    #    log10Teff_opt, log10rdstar_opt = result["x"]
    
    #    Teff_opt = 10.**log10Teff_opt
    #    rdstar_opt = 10.**log10rdstar_opt
    
        #print Teff_opt, rdstar_opt


    #use_models = False
    
    #if math.isnan(Teff_opt) or math.isnan(rdstar_opt) or (rdstar_opt == 0.0):
    #    print 'Fitting failed'
    #    #plt.errorbar(cent_wavs_full_use, nuFnu_full_use, yerr=nuFnu_full_err_use)
    #    #plt.xscale('log', nonposx='clip')
    #    #plt.yscale('log', nonposy='clip')
    #    #plt.show()
        
    #    subj.fitfail = True
        
    #    return subj
    
    nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)        

    use_next = ((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) < 10.
    
    extra_check = False
    
    if Teff_opt < 5000.:
        subj.use_models = True
    elif Teff_opt < 7000. and not use_next:
        subj.use_models = True
    elif Teff_opt > 5000. and Teff_opt < 7000. and use_next:
        filter_cut += 1
        #print "W1 significantly low. Refitting with W2 included"
        #popt, pcov = curve_fit(star_fitter_blackbody, subj.centwavs_meters[:-2], subj.nuFnu[:-2], sigma = subj.nuFnuerrs[:-2], p0 = [popt[0], popt[1]])
        #print len(cent_wavs_full_use), len(nuFnu_full_use), len(nuFnu_full_err_use)
        result = minimize(nll, [log10Teff_opt, log10rdstar_opt], args = (cent_wavs_full_use[:filter_cut], nuFnu_full_use[:filter_cut], nuFnu_full_err_use[:filter_cut]))
        log10Teff_opt_new, log10rdstar_opt_new = result["x"]
        #print 10.**log10Teff_opt_new, 10.**log10rdstar_opt_new
        if (10.**log10Teff_opt) < 7000.:
            subj.use_models = True
            extra_check = True
    
    #Teff_opt = popt[0]
    #rdstar_opt = popt[1]
    
    model_star_plotting = False
    
    if subj.use_models:
        cent_wavs_fit = cent_wavs_full_use[:filter_cut]
        mags_fit = mags_full_use[:filter_cut]
        magerrs_fit = magerrs_full_use[:filter_cut]
        #print "W1 significantly low. Refitting with W1 included"

        #else:
        #    cent_wavs_fit = cent_wavs_full_use[:-4]
        #    mags_fit = mags_full_use[:-4]
        #    magerrs_fit = magerrs_full_use[:-4]
        Teff_use, logg_use, rdstar_use = star_fitter_models(cent_wavs_fit, mags_fit, magerrs_fit, [log10Teff_opt, log10rdstar_opt])

        #print Teff_use, logg_use, rdstar_use
        
        model_mags = btsettl_models_dict[(Teff_use, logg_use)][:-3]
        
        model_mags_at_d = model_mags - 5.*np.log10(rdstar_use)
        
        model_mags_full = btsettl_models_dict[(Teff_use, logg_use)]
        model_mags_full_at_d = model_mags_full - (5.*np.log10(rdstar_use))
        print len(model_mags_full_at_d)

        #print model_mags_full_at_d
        #print mags_full_use
        
        chi2model = np.sum(np.array([((mags_fit[i] - model_mags_at_d[i])**2) for i in range(len(mags_fit))]) / magerrs_fit) / len(mags_fit)
        
        fluxes_at_d = np.zeros(mags_full_use.size)
        
        #print (subj.has_tycho_vmag or subj.has_tycho_bmag)
        
        for i in range(fluxes_at_d.size):
            #print cent_wavs_full_use[i]
            #print cent_wavs_full_use[i]
            index = cent_wavs_dict[cent_wavs_full_use[i]]
            print index, btsettl_column_labels[2+index], cent_wavs_dict_keys[index], filterzps_full_use[i]
            #print index
            
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
                #print "W2 significantly low. Refitting with W2 included"

                Teff_use, logg_use, rdstar_use = star_fitter_models(cent_wavs_refit, mags_refit, magerrs_refit, [log10Teff_opt, log10rdstar_opt])

                #print Teff_use, logg_use, rdstar_use
        
                model_mags_refit = btsettl_models_dict[(Teff_use, logg_use)][:-2]
        
                model_mags_refit_at_d = model_mags_refit - 5.*np.log10(rdstar_use)
        
                model_mags_refit_full = btsettl_models_dict[(Teff_use, logg_use)]
                model_mags_refit_full_at_d = model_mags_refit_full - (5.*np.log10(rdstar_use))

                #print model_mags_full_at_d
                #print mags_full_use
        
                #chi2model = np.sum(np.array([((mags_fit[i] - model_mags_at_d[i])**2) for i in range(len(mags_fit))]) / magerrs_fit) / len(mags_fit)
        
                fluxes_at_d = np.zeros(mags_full_use.size)
        
                for i in range(fluxes_at_d.size):
                    #print cent_wavs_full_use[i]
                    #print cent_wavs_full_use[i]
                    index = cent_wavs_dict[cent_wavs_full_use[i]]
            
                    fluxes_at_d[i] = filterzps_full_use[i] * (10.**(-0.4*model_mags_refit_full_at_d[index])) * (10.**(-23.))
                
                nuFnu_model = (c / cent_wavs_full_use) * fluxes_at_d
        
                nuFnu_remain = nuFnu_full_use - nuFnu_model
            
                print filter_cut, nuFnu_full_use[filter_cut], nuFnu_model[filter_cut], nuFnu_full_err_use[filter_cut], ((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut])
            
                #if (((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) or (filter_cut > -3):
                if ((((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) and filter_cut > -4) or (((nuFnu_full_use[filter_cut] - nuFnu_model[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 10.) or (filter_cut > -3):
                    use_next = False

        subj.log10Teffguess = np.log10(Teff_use)
        subj.Teffguess = Teff_use
        subj.logg_guess = logg_use
        subj.log10rdstarguess = np.log10(rdstar_use)
        subj.filter_cut = filter_cut
            
            #if not use_w2:
            #    cent_wavs_fit_short = cent_wavs_full_use[:-4]
            #    mags_fit_short = mags_full_use[:-4]
            #    magerrs_fit_short = magerrs_full_use[:-4]
            
            #    print Teff_opt
            
            #    Teff_use_short, logg_use_short, rdstar_use_short = star_fitter_models(cent_wavs_fit_short, mags_fit_short, magerrs_fit_short, [np.log10(Teff_opt), np.log10(rdstar_opt)])
            
            #    model_mags_short = np.array([btsettl_models_dict[(Teff_use_short, logg_use_short)][cent_wavs_dict[x]] for x in cent_wavs_fit_short])
                #for wav in cent_wavs_fit_short:
                
                #model_mags_short = btsettl_models_dict[(Teff_use_short, logg_use_short)][i][cent_wavs_dict[cent_wavs_fit_short]]
            #    model_mags_short_at_d = model_mags_short - 5.*np.log10(rdstar_use_short)
            
            #    print len(cent_wavs_fit_short)
            #    print len(mags_fit_short)
            #    print len(magerrs_fit_short)
            
            
            #    print len(mags_fit_short), len(model_mags_short_at_d), len(magerrs_fit_short)
            
            #    chi2short = np.sum(np.square(mags_fit_short - model_mags_short_at_d)/magerrs_fit_short) / len(mags_fit_short)
                #chi2testshort = np.sum(np.square(nuFnu_use - nuFnu_star_test_short)/nuFnu_err_use) / len(nuFnu_use)
            
            #    if chi2short < chi2model:
            #        print '2MASS only has better chi2'
            #        subj.Teff = Teff_use_short
        #        subj.logg = logg_use_short
        #        subj.rdstar = rdstar_use_short
        #    else:
        #        subj.Teff = Teff_use
        #        subj.logg = logg_use
        #        subj.rdstar = rdstar_use
        #else:
        #    subj.Teff = Teff_use
        #    subj.logg = logg_use
        #    subj.rdstar = rdstar_use
            
        subj.nuFnu_star = nuFnu_model
            
        #print 'nuFnu_star', subj.nuFnu_star.size
        
        teffpull = '0'+str(int(Teff_use/100))
        loggpull = str(logg_use)
        
        spect_file = '/discover/nobackup/ssilverb/for_discover/BT-Settl_M-0.0a+0.0/lte'+teffpull+'.0-'+loggpull+'-0.0a+0.0.BT-Settl.spec.7'
        
        spec_X, spec_S, spec_dict = fluxdrive_plot(spect_file,1)
        
        #for i in range(spec_X.size):
        #    if spec_X[i] > 2500. and spec_X[i] < 50000.:
        #        print spec_X[i], spec_S[i]
        
        flux_spec_S = spec_X * spec_S * (rdstar_use**2)
        
        #spec_S_interp = np.interp(plotting_xvec_angstroms, spec_X, spec_S)
        
        
        #print 2208. * spec_S_adjust_flux * rdstar_use
        #print spec_S_adjust_flux
        #print subj.nuFnu_star[-1]
        
        #print spec_S_at_bands
        #print subj.nuFnu_star
        
        #print plotting_xvec_angstroms
        #print spec_S_interp
        
        #for i in range(spec_S_interp.size):
        #    print plotting_xvec_angstroms[i], spec_S_interp[i]
        
        #print subj.nuFnu_star[-1]/spec_S_adjust_flux
        
        subj.nuFnu_star_plotting_temp = np.interp(plotting_xvec_angstroms, spec_X, flux_spec_S)
        model_star_plotting = True
        
        
    else:
        if use_next:
            while use_next:
                print subj.filts_to_use[filter_cut], 'not in excess. Refitting with', subj.filts_to_use[filter_cut], 'included.'
                #print "W1 significantly low. Refitting with W1 included"
                filter_cut += 1
                
                result = minimize(nll, [log10Teff_opt, log10rdstar_opt], args = (cent_wavs_full_use[:filter_cut], nuFnu_full_use[:filter_cut], nuFnu_full_err_use[:filter_cut]))
                #print subj.
    
                log10Teff_opt, log10rdstar_opt = result["x"]
                
                Teff_opt = 10.**log10Teff_opt
                rdstar_opt = 10.**log10rdstar_opt

                #print Teff_opt, rdstar_opt
            
                #popt, pcov = curve_fit(star_fitter_blackbody, subj.centwavs_meters[:-2], subj.nuFnu[:-2], sigma = subj.nuFnuerrs[:-2], p0 = [popt[0], popt[1]])

                #subj.Teff = Teff_opt
                #subj.Teff_err = (pcov[0,0])**2
                #subj.rdstar = rdstar_opt
                #subj.rdstar_err = (pcov[1,1])**2
        
                #print subj.Teff, subj.rdstar
                nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt**2) * u.cm * u.cm * u.s / u.erg)

                nuFnu_star_test = nuFnu_star[:filter_cut]
    
                #chi2test = np.sum(np.square(nuFnu_use - nuFnu_star_test)/nuFnu_err_use) / len(nuFnu_use)
    
                nuFnu_remain = nuFnu_full_use - nuFnu_star
        
                #print nuFnu_remain[filter_cut]
                #print nuFnu_full_use[filter_cut] - (3.*(nuFnu_full_err_use[filter_cut]))
                #print filter_cut
    
                print filter_cut, nuFnu_full_use[filter_cut], nuFnu_star[filter_cut], nuFnu_full_err_use[filter_cut], ((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut])
                if ((((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) and filter_cut > -4) or (((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 10.) or (filter_cut > -3):
                    use_next = False
                #if (((nuFnu_full_use[filter_cut] - nuFnu_star[filter_cut]) / nuFnu_full_err_use[filter_cut]) > 5.) or (filter_cut > -3):
                #    use_next = False
                
            subj.log10Teffguess = np.log10(Teff_opt)
            subj.log10rdstarguess = np.log10(rdstar_opt)   

            
            #if use_w2:
                #print "W2 significantly low. Refitting with W2 included"
            
            #    result = minimize(nll, [log10Teff_opt, log10rdstar_opt], args = (cent_wavs_full_use[:-2], nuFnu_full_use[:-2], nuFnu_full_err_use[:-2]))
                #print subj.
    
            #    log10Teff_opt, log10rdstar_opt = result["x"]
    
            #    Teff_opt = 10.**log10Teff_opt
            #    rdstar_opt = 10.**log10rdstar_opt

                #print Teff_opt, rdstar_opt
            
                #popt, pcov = curve_fit(star_fitter_blackbody, subj.centwavs_meters[:-2], subj.nuFnu[:-2], sigma = subj.nuFnuerrs[:-2], p0 = [popt[0], popt[1]])

            #    subj.log10Teffguess = np.log10(Teff_opt)
                #subj.Teff_err = (pcov[0,0])**2
            #    subj.log10rdstarguess = np.log10(rdstar_opt)
                #subj.rdstar_err = (pcov[1,1])**2
        
                #print subj.Teff, subj.rdstar
            
        else:
            #nuFnu_shortened = nuFnu_full_use[:-4]
            #nuFnu_err_shortened = nuFnu_full_err_use[:-4]
            #cent_wavs_shortened = cent_wavs_full_use[:-4]

            #logTeff_guess = np.log10(Teff_opt)
            #logrdstar_guess = np.log10(rdstar_opt)
    
            #print nuFnu_use

            #result = minimize(nll, [logTeff_guess, logrdstar_guess], args = (cent_wavs_shortened, nuFnu_shortened, nuFnu_err_shortened))
    
            #log10Teff_opt_short, log10rdstar_opt_short = result["x"]
    
            #Teff_opt_short = 10.**log10Teff_opt_short
            #rdstar_opt_short = 10.**log10rdstar_opt_short
        
            #popt, pcov = curve_fit(star_fitter_blackbody, cent_wavs_use, nuFnu_use, sigma = nuFnu_err_use, p0 = [logTeff_guess, logrdstar_guess])
            #Teff_opt_short = 10.**(popt[0])
            #rdstar_opt_short = 10.**(popt[1])

            #chi2test = 
    
            #nuFnu_star = np.array(blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, Teff_opt_short * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * (rdstar_opt_short**2) * u.cm * u.cm * u.s / u.erg)

            #nuFnu_star_test_short = nuFnu_star[:-4]
    
            #chi2testshort = np.sum(np.square(nuFnu_shortened - nuFnu_star_test_short)/nuFnu_err_shortened) / len(nuFnu_shortened)

            #if chi2testshort < chi2test:
            #    print "2MASS only has better chi2"
            #    subj.Teff = Teff_opt_short
            #    subj.rdstar = rdstar_opt_short
            #else:
            subj.log10Teffguess = np.log10(Teff_opt)
            subj.log10rdstarguess = np.log10(rdstar_opt)
            subj.filter_cut = filter_cut
                
        if not model_star_plotting:
            #print len(cent_wavs_full_use)
            #print subj.Teff
            #print subj.rdstar
            subj.nuFnu_star = blackbody_lambda(cent_wavs_full_use * 1.e10 * u.AA, (10.**subj.log10Teffguess) * u.K) * np.pi * u.sr * (cent_wavs_full_use * 1.e10 * u.AA) * ((10.**subj.log10rdstarguess)**2) * u.cm * u.cm * u.s / u.erg
        
        #print nuFnu_use
        #print subj.nuFnu_star
       
        #print subj.nuFnu
        #print subj.nuFnu_star
        if not model_star_plotting:    
            subj.nuFnu_star_plotting_temp = blackbody_lambda(plotting_xvec_angstroms * u.AA, (10.**subj.log10Teffguess)*u.K) * np.pi * u.sr * (plotting_xvec_angstroms * u.AA) * ((10.**subj.log10rdstarguess)**2) * u.cm * u.cm * u.s / u.erg
    
    #test_subj = get_star_fit_2mass(subj)
    
    #if not subj.use_gaia:
    #    subj.nuFnu_disk = subj.nuFnu_nogaia - subj.nuFnu_star
    #    subj.nuFnu_disk_errs = subj.nuFnuerrs_nogaia
    #else:
    #    subj.nuFnu_disk = subj.nuFnu_nops - subj.nuFnu_star
    #    subj.nuFnu_disk_errs = subj.nuFnuerrs_nops
     
    subj.nuFnu_disk = nuFnu_full_use - subj.nuFnu_star
    subj.nuFnu_disk_errs = nuFnu_full_err_use
                    
    #if k_excess:
    #    boundary_cond = -5
    #elif use_w2:
    #    boundary_cond = -2
    #elif use_w1:
    #    boundary_cond = -3
    #else:
    #    boundary_cond = -4
        
    #print subj.nuFnu_disk[:boundary_cond]
    #print subj.nuFnu_disk_errs[:boundary_cond]
    
    #test_fit = subj.nuFnu_star[:boundary_cond] / (subj.nuFnu_star[:boundary_cond] + subj.nuFnu_disk[:boundary_cond])
    test_fit = subj.nuFnu_disk[:filter_cut] / subj.nuFnu_disk_errs[:filter_cut]
     
    test_fit_matches = 0
    
    for i in range(len(test_fit)):
        #print subj.nuFnu_star[i], (subj.nuFnu_star[i] + subj.nuFnu_disk[i])
        #print test_fit[i]
        if np.abs(test_fit[i]) < 5.:
            test_fit_matches += 1
    
    #print test_fit
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
        
    #print subj.good_star_fit
    
    #subj.nuFnu_disk_errs = subj.nuFnuerrs
    
    
    subj.sig_disk = subj.nuFnu_disk[-4:] / subj.nuFnu_disk_errs[-4:]
    
    subj.num_excesses = subj.sig_disk[subj.sig_disk > 5].size
    
    if subj.num_excesses < 1:
        subj.num_excesses = subj.sig_disk[subj.sig_disk > 3].size
    
    return subj

start_time = time.time()
subjs_with_star_fits = []

print start_time

#f0 = open('has_no_fit.csv','w')
#f1 = open('has_w3_excess.csv','w')
#f2 = open('has_no_excess.csv','w')
#f3 = open('has_w4_excess.csv','w')
#f4 = open('bad_star_fits.csv','w')
#f5 = open('has_more_than_two_excesses.csv','w')

#f0 = open('excesses_for_each_object.csv','w')

count_good_star_fits = 0
count_good_star_model_fits = 0
count_good_star_bb_fits = 0

count_bb_fits = 0
count_model_fits = 0

for subj in raw_subjs:
    print subj.zooniverse_id, subj.wiseid
    subjs_with_star_fits.append(get_star_fit(subj))
    
    if subj.log10Teffguess is not None:
        print 10.**subj.log10Teffguess, subj.logg_guess, 10.**subj.log10rdstarguess
    else:
        print subj.log10Teffguess, subj.logg_guess, subj.log10rdstarguess
    
    if subj.use_models:
        count_model_fits += 1
    else:
        count_bb_fits += 1

    if subj.fitfail:
        print 'Fail'
        #f0.write(subj.zooniverse_id+','+subj.wiseid+'\n')
    #elif not subj.good_star_fit:
        #f4.write(subj.zooniverse_id+','+subj.wiseid+'\n')
    else:
        count_good_star_fits += 1
        
        if subj.use_models:
            count_good_star_model_fits += 1
        else:
            count_good_star_bb_fits += 1
            
        #if subj.num_excesses > 2:
            #f5.write(subj.zooniverse_id+','+subj.wiseid+'\n')
        #elif subj.num_excesses > 1:
            #f1.write(subj.zooniverse_id+','+subj.wiseid+'\n')
            #count_good_star_fits += 1
        #elif subj.num_excesses < 1:
            #f2.write(subj.zooniverse_id+','+subj.wiseid+'\n')
            #count_good_star_fits += 1
        #else:
            #f3.write(subj.zooniverse_id+','+subj.wiseid+'\n')
            #count_good_star_fits += 1

    #plt.rcParams['xtick.labelsize'] = 15
    #plt.rcParams['ytick.labelsize'] = 15

    #if subj.has_tycho_vmag or subj.has_tycho_bmag:
    #    plt.errorbar(subj.centwavs_microns_to_use, subj.nuFnu_to_use, yerr=subj.nuFnuerrs_to_use, fmt='k.')
    #else:
    #plt.errorbar(subj.centwavs_microns_nops, subj.nuFnu_nops, yerr=subj.nuFnuerrs_nops, fmt='k.')
    #if not subj.fitfail:
        #plt.plot(plotting_x_vec, subj.nuFnu_star_plotting_temp)
        #if subj.has_tycho_vmag or subj.has_tycho_bmag:
        #    plt.plot(subj.centwavs_microns_nogaia, subj.nuFnu_star, 'o')
        #else:
        #plt.plot(subj.centwavs_microns_to_use, subj.nuFnu_star, 'o')
    
    #ymin = float(np.floor((np.log10(min(subj.nuFnu_nops))-0.5)*2.))/2.
    #ymax = float(np.ceil(2.*(np.log10(max(subj.nuFnu_nops))+0.5)))/2.
    
    #plt.xlabel(r'$\mathrm{Wavelength(\mu m)}$', fontsize=20)
    #plt.ylabel(r'$\mathrm{\nu F_{\nu} (erg\,s^{-1}\,cm^{-2})}$', fontsize=20)


    
    #ymin = min(subj.nuFnu_star_plotting_temp)
    #ymax = max(subj.nuFnu_star_flotting_temp)
    
    
    
    #plt.xscale("log", nonposx='clip')
    #plt.yscale("log", nonposy='clip')
    #plt.xlim([0.25, 50.])
    #plt.ylim([10.**ymin, 10.**ymax])
    #plt.show()
    
    #f0.write(subj.zooniverse_id+','+subj.wiseid+','+str(subj.num_excesses)+'\n')
    
    print time.time(), len(subjs_with_star_fits), (time.time() - start_time)/len(subjs_with_star_fits)
    print ''
    
#f0.close()
star_fit_end_time = time.time()

#f0.close()
#f1.close()
#f2.close()
#f3.close()
#f4.close()
#f5.close()

#winsound.Beep(440,1000)
print "Star fits done"
print "Time elapsed:", star_fit_end_time - start_time
print "Number good star fits:", count_good_star_fits, "out of 242"
print "Number good model fits:", count_good_star_model_fits, "out of", count_model_fits
print "Number good bb fits:", count_good_star_bb_fits, "out of", count_bb_fits

def plot_subj_temp(subj):
    plt.figure()
    
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
        
    plt.gcf().subplots_adjust(left = 0.18)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)
    plt.gcf().subplots_adjust(bottom=0.16)
    
    #ax = plt.subplot
    
    plt.errorbar(subj.centwavs_microns_to_use, subj.nuFnu_to_use, yerr=subj.nuFnuerrs_to_use, fmt='k.')
        
    #plt.plot(subj.centwavs_microns, subj.nuFnu_star, 'r.')
        
    #print subj.nuFnu
    #print subj.nuFnu_star
    

    plt.xscale('log', nonposx='clip')
    plt.yscale('log', nonposy='clip')
    
    min_nuFnu_logs = min(np.log10(subj.nuFnu))
    max_nuFnu_logs = max(np.log10(subj.nuFnu_star_plotting_temp))

    ymin = float(np.floor((min_nuFnu_logs-1.5)*2.))/2.
    ymax = float(np.ceil(2.*(max_nuFnu_logs+0.5)))/2.

    #if subj.good_disk_fit:
    plt.plot(plotting_x_vec, subj.nuFnu_star_plotting_temp, linestyle=':', label='Stellar fit')
    plt.plot(plotting_x_vec, subj.nuFnu_disk_plotting_temp_blackbody, linestyle='--', label='Disk blackbody')
    plt.plot(plotting_x_vec, subj.nuFnu_model_plotting_temp_blackbody, label='Combined blackbody model')

    if subj.num_excesses > 2:
        plt.plot(plotting_x_vec, subj.nuFnu_disk_plotting_temp_powerlaw, linestyle='--', label='Disk powerlaw')
        plt.plot(plotting_x_vec, subj.nuFnu_model_plotting_temp_powerlaw, label='Combined powerlaw model')

    
    #plt.annotate(subj.wiseid+'\n'+'Tstar = %.0f K\nTdisk = %.0f K\nLir/Lstar = %.4e' % (subj.Teff, subj.Tdisk, subj.fir), xy=(0.55, 0.75), xycoords = 'axes fraction', fontsize=12)

    #else:
    #plt.plot(plotting_x_vec, subj.nuFnu_model_plotting_temp)
    #plt.annotate(subj.wiseid+'\n'+'Tstar = %.0f K\nNo good disk fit' % (subj.Teff), xy=(0.55, 0.75), xycoords = 'axes fraction', fontsize=12)
        
    plt.legend(loc='lower left',fontsize=12)
    
    
    plt.xlim([0.25, 100.])
    plt.ylim([10.**(ymin), 10.**(ymax)])
    
    plt.xlabel(r'$\mathrm{Wavelength(\mu m)}$', fontsize=20)
    plt.ylabel(r'$\mathrm{\nu F_{\nu} (erg\,s^{-1}\,cm^{-2})}$', fontsize=20)
    
    #if len(subjs_with_star_fits) < 2:
    #plt.savefig(subj.zooniverse_id+'_modelfit.pdf')
    
    #plt.close()
    
    plt.show()
    return

def plot_subj(subj):
    plt.figure()
    
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
        
    plt.gcf().subplots_adjust(left = 0.18)
    plt.gcf().subplots_adjust(right=0.95)
    plt.gcf().subplots_adjust(top=0.95)
    plt.gcf().subplots_adjust(bottom=0.16)
    
    #ax = plt.subplot
    
    plt.errorbar(subj.centwavs_microns_nops, subj.nuFnu_nops, yerr=subj.nuFnuerrs_nops, fmt='k.')
        
    #plt.plot(subj.centwavs_microns, subj.nuFnu_star, 'r.')
        
    #print subj.nuFnu
    #print subj.nuFnu_star
    

    plt.xscale('log', nonposx='clip')
    plt.yscale('log', nonposy='clip')
    
    min_nuFnu_logs = min(np.log10(subj.nuFnu))
    max_nuFnu_logs = max(np.log10(subj.nuFnu_star_plotting))

    ymin = float(np.floor((min_nuFnu_logs-1.5)*2.))/2.
    ymax = float(np.ceil(2.*(max_nuFnu_logs+0.5)))/2.

    #if subj.good_disk_fit:
    plt.plot(plotting_x_vec, subj.nuFnu_star_plotting, linestyle=':', label='Stellar fit')
    plt.plot(plotting_x_vec, subj.nuFnu_disk_plotting, linestyle='--', label='Disk blackbody')
    plt.plot(plotting_x_vec, subj.nuFnu_model_plotting, label='Combined model')
    #plt.annotate(subj.wiseid+'\n'+'Tstar = %.0f K\nTdisk = %.0f K\nLir/Lstar = %.4e' % (subj.Teff, subj.Tdisk, subj.fir), xy=(0.55, 0.75), xycoords = 'axes fraction', fontsize=12)

    #else:
    #    plt.plot(plotting_x_vec, subj.nuFnu_model_plotting)
    #    #plt.annotate(subj.wiseid+'\n'+'Tstar = %.0f K\nNo good disk fit' % (subj.Teff), xy=(0.55, 0.75), xycoords = 'axes fraction', fontsize=12)
        
    plt.legend(loc='lower left',fontsize=12)
    
    
    plt.xlim([0.25, 100.])
    plt.ylim([10.**(ymin), 10.**(ymax)])
    
    plt.xlabel(r'$\mathrm{Wavelength(\mu m)}$', fontsize=20)
    plt.ylabel(r'$\mathrm{\nu F_{\nu} (erg\,s^{-1}\,cm^{-2})}$', fontsize=20)
    
    #if len(subjs_with_star_fits) < 2:
    #plt.savefig(subj.zooniverse_id+'_modelfit.pdf')
    
    #plt.close()
    
    plt.show()
    return
	
def lnlike_models(theta, centwavs, nuFnu, nuFnuerrs, Teff, logg):
    log10rdstar, log10Tdisk, log10xdisk = theta
    
    #Teff = 10.**log10Teff
    rdstar = 10.**log10rdstar
    Tdisk = 10.**log10Tdisk
    xdisk = 10.**log10xdisk
    
    #Teff_use = round(Teff)
    
    #if 0.673*1.e6 in centwavs:
    #    filterzps_use = filterzps_nops
    #else:
    #    filterzps_use = filterzps_nogaia
    
    #centwavs_angstroms_num = centwavs * 1.e10
    centwavs_angstroms = centwavs * 1.e10 * u.AA
    #mags_use = np.array(mags)
                
    #magerrs_use = np.array(magerrs)
                
    fitmagsfull = btsettl_models_dict[(Teff, logg)]
                
    fitmagsuse = []
    filterzps_use = []
                
    #print centwavs
                
    for i in range(len(centwavs)):
        #print cent_wavs_dict[centwavs[i]]
        #print cent_wavs_dict[centwavs[i]]
        wavtemp = centwavs[i] * 1.e6
        if wavtemp > 1.:
            wav = round(wavtemp,3)
        else:
            wav = round(wavtemp,4)
        fitmagsuse.append(fitmagsfull[cent_wavs_dict[centwavs[i]]])
        filterzps_use.append(filterzps_dict[wav])
        
        #if centwavs[i] == cent_wavs_dict_keys[0]:
        #    magerrs_use[i] = 10.*magerrs[i]

    #print fitmagsuse
                
    result_mags = fitmagsuse - 5.*log10rdstar
    
    fluxes_at_d = np.zeros(nuFnu.size)
    #fluxerrs_at_d = np.zeros(nuFnu.size)
        
    for i in range(fluxes_at_d.size):
        fluxes_at_d[i] = filterzps_use[i] * 10.**(-0.4*result_mags[i]) * 10.**(-23.)
        #fluxerrs_at_d[i] = 0.4 * fluxes_at_d[i] * np.log(10) * magerrs[i]
            
    nuFnu_star_model = (c / centwavs) * fluxes_at_d
    #nuFnu_errs_star_model = (c / centwavs) * fluxerrs_at_d
    
    flux_disk_lam_temp = blackbody_lambda(centwavs_angstroms, Tdisk*u.K) * np.pi * u.sr
    
    flux_disk_lam = flux_disk_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg
    
    disk_model = flux_disk_lam * xdisk
    
    model = nuFnu_star_model + disk_model        
                
    inv_sigma2 = 1./(nuFnuerrs**2)
                
    return -0.5*np.sum((((nuFnu - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))

nllmodelsblackbody = lambda *args: -lnlike_models(*args)

    
def lnlike_blackbody_powerlaw(theta, centwavs, nuFnu, nuFnuerrs):
    log10Teff, log10rdstar, alpha, log10beta = theta
    
    #Teff = 10.**log10Teff
    #rdstar = 10.**log10rdstar
    #Tdisk = 10.**log10Tdisk
    #xdisk = 10.**log10xdisk
    
    centwavs_angstroms = centwavs * 1.e10 * u.AA
    log10centwavs = np.log10(centwavs)
    #centwavs_meters = centwavs * u.m
    
    #flux_bb_lam_temp = 
    
    #flux_bb_lam = flux_bb_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg

    bb_model = blackbody_lambda(centwavs_angstroms, (10.**log10Teff)*u.K) * np.pi * u.sr * centwavs_angstroms * u.cm * u.cm * u.s / u.erg * ((10.**log10rdstar)**2)
    
    #print Teff, rdstar
    #print "flux:", flux
    #print "model:", model
    
    #flux_disk_lam_temp = 
    
    #flux_disk_lam = 
    
    #disk_model = blackbody_lambda(centwavs_angstroms, (10.**log10Tdisk)*u.K) * np.pi * u.sr * centwavs_angstroms * u.cm * u.cm * u.s / u.erg * (10.**log10xdisk)
    
    #log10disklamFlam = alpha*log10centwavs[-5:] + beta
    
    #bb_cut_position = blackbody_lambda((2.e4*u.AA), (10.**log10Teff)*u.K) * np.pi * u.sr * centwavs_angstroms * u.cm * u.cm * u.s / u.erg * ((10.**log10rdstar)**2)
    
    #centwavs_diskfit = np.zeros(6)
    #centwavs_diskfit[0] = 2.e-6
    #centwavs_diskfit[-5:] = centwavs[-5:]
    
    centwavs_diskfit = centwavs[-5:]
    
    diskmodel = (10.**log10beta)*(np.power(np.array(centwavs_diskfit),alpha))
    
    full_disk_model = np.zeros(len(centwavs))
    
    full_disk_model[-5:] = diskmodel
    
    #diskmodel_first = np.zeros(len(centwavs_diskfit))
    
    #diskmodel_first = beta*(centwavs_diskfit**alpha)
    
    #dif = bb_cut_position - diskmodel_first[0]
    
    #diskmodel = diskmodel_first + dif
    
    #diskmodel = np.zeros(len(centwavs))
    #diskmodel[-5:] = 10.**log10disklamFlam

    
    model = np.zeros(len(centwavs))
    
    for i in range(model.size):
        model[i] = max(bb_model[i],full_disk_model[i])
    
    #model[:-5] = bb_model[:-5]
    #model[-5:] = diskmodel[-5:]

    inv_sigma2 = 1./(nuFnuerrs**2)
    
    #print -0.5*np.sum((((flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))
    
    return -0.5*np.sum((((nuFnu - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))

nllblackbodypowerlaw = lambda *args: -lnlike_blackbody_powerlaw(*args)


def lnlike_models_powerlaw(theta, centwavs, nuFnu, nuFnuerrs, Teff, logg):
    log10rdstar, alpha, log10beta = theta
    
    #Teff = 10.**log10Teff
    rdstar = 10.**log10rdstar
    #Tdisk = 10.**log10Tdisk
    #xdisk = 10.**log10xdisk
    
    #Teff_use = round(Teff)
    
    #if 0.673*1.e6 in centwavs:
    #    filterzps_use = filterzps_nops
    #else:
    #    filterzps_use = filterzps_nogaia
    
    #centwavs_angstroms_num = centwavs * 1.e10
    centwavs_angstroms = centwavs * 1.e10 * u.AA
    log10centwavs = np.log10(centwavs)
    #log10nuFnu = np.log10(nuFnu)
    #log10nuFnuerrs = (1./np.log(10.)) * (nuFnuerrs/nuFnu)
    
    #mags_use = np.array(mags)
                
    #magerrs_use = np.array(magerrs)
                
    fitmagsfull = btsettl_models_dict[(Teff, logg)]
                
    fitmagsuse = []
    filterzps_use = []
                
    #print centwavs
                
    for i in range(len(centwavs)):
        #print cent_wavs_dict[centwavs[i]]
        #print cent_wavs_dict[centwavs[i]]
        wavtemp = centwavs[i] * 1.e6
        if wavtemp > 1.:
            wav = round(wavtemp,3)
        else:
            wav = round(wavtemp,4)
        fitmagsuse.append(fitmagsfull[cent_wavs_dict[centwavs[i]]])
        filterzps_use.append(filterzps_dict[wav])
        
        #if centwavs[i] == cent_wavs_dict_keys[0]:
        #    magerrs_use[i] = 10.*magerrs[i]

    #print fitmagsuse
                
    result_mags = fitmagsuse - 5.*log10rdstar
    
    fluxes_at_d = np.zeros(nuFnu.size)
    #fluxerrs_at_d = np.zeros(nuFnu.size)
        
    for i in range(fluxes_at_d.size):
        fluxes_at_d[i] = filterzps_use[i] * 10.**(-0.4*result_mags[i]) * 10.**(-23.)
        #fluxerrs_at_d[i] = 0.4 * fluxes_at_d[i] * np.log(10) * magerrs[i]
            
    nuFnu_star_model = (c / centwavs) * fluxes_at_d
    #nuFnu_errs_star_model = (c / centwavs) * fluxerrs_at_d
    
    
    teffpull = '0'+str(int(Teff/100))
    loggpull = str(logg)
        
    spect_file = '/discover/nobackup/ssilverb/for_discover/BT-Settl_M-0.0a+0.0/lte'+teffpull+'.0-'+loggpull+'-0.0a+0.0.BT-Settl.spec.7'
        
    spec_X, spec_S, spec_dict = fluxdrive_plot(spect_file,1)
    
    flux_spec_S = spec_X * spec_S * (rdstar**2)
    
    #flux_spec_S_use = np.interp(2.e4, spec_X, flux_spec_S)
    
    
    #log10disklamFlam = alpha*log10centwavs[-5:] + beta
    
    disk_centwavs = centwavs[-5:]
    
    disklamFlam_test = (10.**log10beta)*(np.power(np.array(disk_centwavs),alpha))
    
    disk_model = np.zeros(len(centwavs))
    disk_model[-5:] = disklamFlam_test
    
    model = np.zeros(len(centwavs))
    
    for i in range(len(centwavs)):
        model[i] = max(nuFnu_star_model[i],disk_model[i])
    
    #flux_disk_lam_temp = blackbody_lambda(centwavs_angstroms, Tdisk*u.K) * np.pi * u.sr
    
    #flux_disk_lam = flux_disk_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg
    
    #disk_model = flux_disk_lam * xdisk
    
    #model = nuFnu_star_model + disk_model        
                
    inv_sigma2 = 1./(nuFnuerrs**2)
                
    return -0.5*np.sum((((nuFnu - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))

nllmodelspowerlaw = lambda *args: -lnlike_models_powerlaw(*args)
    
    
def lnlike_blackbody(theta, centwavs, nuFnu, nuFnuerrs):
    log10Teff, log10rdstar, log10Tdisk, log10xdisk = theta
    
    #Teff = 10.**log10Teff
    #rdstar = 10.**log10rdstar
    #Tdisk = 10.**log10Tdisk
    #xdisk = 10.**log10xdisk
    
    centwavs_angstroms = centwavs * 1.e10 * u.AA
    #centwavs_meters = centwavs * u.m
    
    #flux_bb_lam_temp = 
    
    #flux_bb_lam = flux_bb_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg

    bb_model = blackbody_lambda(centwavs_angstroms, (10.**log10Teff)*u.K) * np.pi * u.sr * centwavs_angstroms * u.cm * u.cm * u.s / u.erg * ((10.**log10rdstar)**2)
    
    #print Teff, rdstar
    #print "flux:", flux
    #print "model:", model
    
    #flux_disk_lam_temp = 
    
    #flux_disk_lam = 
    
    disk_model = blackbody_lambda(centwavs_angstroms, (10.**log10Tdisk)*u.K) * np.pi * u.sr * centwavs_angstroms * u.cm * u.cm * u.s / u.erg * (10.**log10xdisk)
    
    model = bb_model + disk_model

    inv_sigma2 = 1./(nuFnuerrs**2)
    
    #print -0.5*np.sum((((flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))
    
    return -0.5*np.sum((((nuFnu - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))

nllblackbody = lambda *args: -lnlike_blackbody(*args)

wien_lamFlam_x = 3.90269

wien_lamFlam_factor = (h*c)/(wien_lamFlam_x*k)


def lnlike_disk_blackbody(theta, wav, flux, fluxerr):
    log10Tdisk, log10xdisk = theta

    Tdisk = 10.**log10Tdisk
    xdisk = 10.**log10xdisk
    
    centwavs_angstroms = wav * 1.e10 * u.AA
    centwavs_meters = wav * u.m
    
    flux_lam_temp = blackbody_lambda(centwavs_angstroms, Tdisk*u.K) * np.pi * u.sr
    
    flux_lam = flux_lam_temp * centwavs_angstroms * u.cm * u.cm * u.s / u.erg

    model = flux_lam * xdisk
    
    #print Teff, rdstar
    #print "flux:", flux
    #print "model:", model

    inv_sigma2 = 1./(fluxerr**2)
    
    #print -0.5*np.sum((((flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))
    
    return -0.5*np.sum((((flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))

nll2 = lambda *args: -lnlike_disk_blackbody(*args)


def lnlike_disk_powerlaw(theta, wav, flux, fluxerr):
    alpha, log10beta = theta
    
    log10flux = np.log10(flux)
    log10fluxerr = (1./np.log(10)) * (fluxerr/flux)
    
    log10wav = np.log10(wav)
    
    model = (alpha * log10wav) + log10beta
    
    inv_sigma2 = 1./(log10fluxerr**2)
    
    return -0.5*np.sum((((log10flux - model)**2)*inv_sigma2) + np.log(2.*np.pi) - np.log(inv_sigma2))

nll3 = lambda * args: -lnlike_disk_powerlaw(*args)


def get_disk_fit(subj):
    subj.good_disk_fit = False
    if subj.filter_cut < -4:
        nuFnu_use = subj.nuFnu_disk[subj.filter_cut:]
        nuFnu_err_use = subj.nuFnu_disk_errs[subj.filter_cut:]
        nuFnu_full_use = subj.nuFnu_to_use[subj.filter_cut:]
        nuFnu_full_err_use = subj.nuFnuerrs_to_use[subj.filter_cut:]
        cent_wavs_use = subj.centwavs_meters_to_use[subj.filter_cut:]
    else:
        nuFnu_use = subj.nuFnu_disk[-5:]
        nuFnu_err_use = subj.nuFnu_disk_errs[-5:]
        nuFnu_full_use = subj.nuFnu_to_use[-5:]
        nuFnu_full_err_use = subj.nuFnuerrs_to_use[-5:]
        cent_wavs_use = subj.centwavs_meters_to_use[-5:]
        
    
    Teff_use = 10.**subj.log10Teffguess
    rdstar_use = 10.**subj.log10rdstarguess

    if subj.num_excesses < 1:
        print "No significant excess"
        subj.nuFnu_model_plotting_temp = subj.nuFnu_star_plotting_temp
        return subj
    #elif subj.num_excesses > 1:
    #    #Create new guesses for Tdisk, xdisk
    #    Tdiskguess = 0.05 * subj.Teff
    #    xdiskguess = 5.e-3 * ((subj.Teff/Tdiskguess)**4)*(subj.rdstar**2)
    #    print Tdiskguess, xdiskguess
    #    guess = [np.log10(Tdiskguess), np.log10(xdiskguess)]
        
    #    istart = 0 - subj.num_excesses

    #    print "fitting this:", nuFnu_use[istart:], nuFnu_err_use[istart:]
        
    #    result2 = minimize(nll2, guess, args = (cent_wavs_use[istart:], nuFnu_use[istart:], nuFnu_err_use[istart:]))
        
    #    log10Tdisk_opt, log10xdisk_opt = result2["x"]
    #    #log10xdisk_opt = result2["x"][1]
        
    #    subj.Tdisk = 10.**log10Tdisk_opt
    #    subj.xdisk = 10.**log10xdisk_opt
        
    #    subj.fir = ((subj.Tdisk/subj.Teff)**4)*(subj.xdisk/(subj.rdstar**2))
        
    #    print "Excesses:", subj.num_excesses
    #    nuFnu_disk_points = blackbody_lambda(subj.centwavs_meters * 1.e10 * u.AA, subj.Tdisk * u.K) * np.pi * u.sr * (subj.centwavs_meters * 1.e10 * u.AA) * subj.xdisk * u.cm * u.cm * u.s / u.erg
    #    print "Calc. :", nuFnu_disk_points[-4:]
    #    print "Observed:", subj.nuFnu_disk[-4:]
        
    #    subj.good_disk_fit = True
        
    #    subj.nuFnu_disk_plotting = blackbody_lambda(plotting_xvec_angstroms * u.AA, subj.Tdisk*u.K) * np.pi * u.sr * (plotting_xvec_angstroms * u.AA) * subj.xdisk * u.cm * u.cm * u.s / u.erg
    #    subj.nuFnu_model_plotting = subj.nuFnu_star_plotting  + subj.nuFnu_disk_plotting
        
    #    return subj
    
    else:
        print "Excesses:", subj.num_excesses
        #The hardest case: only one excess to fit to. In this case, we take a grid of possible Tdisk/xdisk combinations, minimize all of them, and see what we find.
        
        #Define a vector of Tdisk guesses
        Tdiskvals = np.linspace(30., 300., 10)
        
        #Also create a dictionary to store xdiskval vectors
        xdiskvals_dict = {x: None for x in Tdiskvals}
        
        #ALSO create dictionaries to store results and negative log likelihoods
        val_chi_dict = {}
        val_results_dict = {}
        
        minimization_start = time.time()
        
        for Tdiskval in Tdiskvals:
            #Define possible xdisk guesses based on the Tdisk guess
            xdiskmax = (rdstar_use**2) * 1.e-2 * ((Teff_use/Tdiskval)**4)
            xdiskmin = (rdstar_use**2) * 1.e-9 * ((Teff_use/Tdiskval)**4)
         
            xdiskvals = np.linspace(xdiskmin, xdiskmax, 11)
            
            xdiskvals_dict[Tdiskval] = xdiskvals    

            for xdiskval in xdiskvals:
                guess = [np.log10(Tdiskval), np.log10(xdiskval)]
                
                #result2 = minimize(nll2, guess, args = (cent_wavs_use, nuFnu_use, nuFnu_err_use))
                
                #log10Tdiskres, log10xdiskres = result2["x"]
                
                #print 
                
                reschi = -lnlike_disk_blackbody(guess, cent_wavs_use, nuFnu_use, nuFnu_err_use)
                
                val_chi_dict[(Tdiskval, xdiskval)] = reschi
                #val_results_dict[(Tdiskval, xdiskval)] = [log10Tdiskres, log10xdiskres]

                #print Tdiskval, xdiskval, log10Tdiskres, log10xdiskres, reschi

        print "Minimization cycle finished. Time:", time.time() - minimization_start
                
        sorted_chis = sorted(val_chi_dict.values())
        sorted_keys = []
        #sorted_results = []
        
        for chi in sorted_chis:
            sorted_keys.append(val_chi_dict.keys()[val_chi_dict.values().index(chi)])
            #sorted_results.append(val_results_dict[sorted_keys[-1]])
            
        num_options = len(sorted_chis)
        
        options_tested = 0
        
        fir_test_res = False
        disk_peak_res = False
        
        while not subj.good_disk_fit and options_tested < num_options:
            key = sorted_keys[options_tested]
            
            guess = [np.log10(key[0]), np.log10(key[1])]
            
            result2 = minimize(nll2, guess, args = (cent_wavs_use, nuFnu_use, nuFnu_err_use))

            log10Tdiskres, log10xdiskres = result2["x"]

            Tdisk_test = 10.**(log10Tdiskres)
            xdisk_test = 10.**(log10xdiskres)
            
            if Tdisk_test == 0.0:
                options_tested += 1
                continue
            
            #print options_tested, Tdisk_test, xdisk_test
            
            tdisk_test_res = Tdisk_test < Teff_use
            
            if max(subj.nuFnu_star_plotting_temp) > 0.:
                peak_star_lamFlam = max(subj.nuFnu_star_plotting_temp)
            else:
                peak_star_lambda = wien_lamFlam_factor / Teff_use
                peak_star_lamFlam = blackbody_lambda(peak_star_lambda * u.m, Teff_use*u.K) * np.pi * u.sr * (peak_star_lambda * u.m) * u.cm * u.cm * u.s / u.erg * (rdstar_use**2)

            
            fir_test = ((Tdisk_test/Teff_use)**4) * (xdisk_test/(rdstar_use**2))
            
            #print fir_test
            
            #print 'Lir/Lstar test:', fir_test
            
            fir_test_res = (fir_test < 1.)
            
            peak_disk_lambda = (wien_lamFlam_factor / Tdisk_test) * 1.e10
            #print peak_disk_lambda
            
            print 'lambda =', peak_disk_lambda, 'Tdisk =', Tdisk_test
            
            peak_disk_lamFlam = blackbody_lambda(peak_disk_lambda * u.AA, Tdisk_test*u.K) * np.pi * u.sr * (peak_disk_lambda * u.AA) * u.cm * u.cm * u.s / u.erg * xdisk_test

            if max(subj.nuFnu_star_plotting_temp) > 0.:
                peak_star_lamFlam = max(subj.nuFnu_star_plotting_temp)
            else:
                peak_star_lambda = wien_lamFlam_factor / Teff_use
                peak_star_lamFlam = blackbody_lambda(peak_star_lambda * u.m, Teff_use*u.K) * np.pi * u.sr * (peak_star_lambda * u.m) * u.cm * u.cm * u.s / u.erg * (rdstar_use**2)

            #print 'Disk peak vs star peak:', peak_disk_lamFlam, peak_star_lamFlam
                
            disk_peak_res = (peak_disk_lamFlam < peak_star_lamFlam)
            
            if (disk_peak_res and fir_test_res) or (subj.nuFnu[-1] > peak_star_lamFlam):
                subj.good_disk_fit = True
                subj.log10Tdiskguess = np.log10(Tdisk_test)
                subj.log10xdiskguess = np.log10(xdisk_test)
                #subj.fir = ((subj.Tdisk/subj.Teff)**4) * (subj.xdisk/(subj.rdstar**2))
                
            options_tested += 1
            #print options_tested
            
        if subj.num_excesses > 2:
            guess = [0., -9]
            result3 = minimize(nll3, guess, args = (cent_wavs_use, nuFnu_full_use, nuFnu_full_err_use))
            
            alpha, log10beta = result3["x"]
            beta = 10.**log10beta
            
            print alpha, beta
            print beta * (cent_wavs_use**alpha)
            print nuFnu_use
            
            powerlaw_component = np.zeros(subj.centwavs_meters_to_use.size)
            
            powerlaw_component[-5:] = ((beta) * (subj.centwavs_meters_to_use[-5:]**alpha))
                
            
            test_model_powerlaw = np.array([max(subj.nuFnu_star[i], powerlaw_component[i]) for i in range(subj.centwavs_meters_to_use.size)])
            test_model_blackbody = subj.nuFnu_star + blackbody_lambda(subj.centwavs_meters_to_use * 1.e10 * u.AA, Tdisk_test*u.K) * np.pi * u.sr * (subj.centwavs_meters_to_use * 1.e10 * u.AA) * u.cm * u.cm * u.s / u.erg * xdisk_test

            if subj.use_models:
                print "using models"
                reschi_powerlaw = lnlike_models_powerlaw([subj.log10rdstarguess, alpha, log10beta], subj.centwavs_meters_to_use, subj.nuFnu_to_use, subj.nuFnuerrs_to_use, subj.Teffguess, subj.logg_guess)
                reschi_blackbody = lnlike_models([subj.log10rdstarguess, subj.log10Tdiskguess, subj.log10xdiskguess], subj.centwavs_meters_to_use, subj.nuFnu_to_use, subj.nuFnuerrs_to_use, subj.Teffguess, subj.logg_guess)
            else:
                print "using blackbody"
                reschi_powerlaw = lnlike_blackbody_powerlaw([subj.log10Teffguess, subj.log10rdstarguess, alpha, log10beta], subj.centwavs_meters_to_use, subj.nuFnu_to_use, subj.nuFnuerrs_to_use)
                reschi_blackbody = lnlike_blackbody([subj.log10Teffguess, subj.log10rdstarguess, subj.log10Tdiskguess, subj.log10xdiskguess], subj.centwavs_meters_to_use, subj.nuFnu_to_use, subj.nuFnuerrs_to_use)
                
            if reschi_powerlaw > reschi_blackbody:
                subj.powerlaw = True
                print "Power law fit superior"
                subj.good_disk_fit = True
                
            
            #test_fit_powerlaw = test_model_powerlaw / subj.nuFnuerrs_to_use
            #test_fit_blackbody = test_model_blackbody / subj.nuFnuerrs_to_use
     
            #test_fit_matches_powerlaw = 0
            #test_fit_matches_blackbody = 0
    
            #for i in range(len(test_fit_powerlaw)):
            #print subj.nuFnu_star[i], (subj.nuFnu_star[i] + subj.nuFnu_disk[i])
            #print test_fit[i]
            #    if np.abs(test_fit_powerlaw[i]) < 10.:
            #        test_fit_matches_powerlaw += 1
                
            #    if np.abs(test_fit_blackbody[i]) < 10.:
            #        test_fit_matches_blackbody += 1
                    
            
            #if test_fit_matches_powerlaw > test_fit_matches_blackbody:
            #    subj.powerlaw = True
            #    print "Power law fit superior"
            #    subj.good_disk_fit = True

            subj.alpha_guess = alpha
            #subj.beta = beta
            subj.log10beta_guess = log10beta
            
            subj.power_law_res = (alpha,beta)

            
        if options_tested == num_options and not subj.good_disk_fit:
            print "Disk fitting failed"
            subj.nuFnu_model_plotting_temp = subj.nuFnu_star_plotting_temp
            return subj
        
        if subj.good_disk_fit:
            if subj.num_excesses > 2:
                nuFnu_disk_points_powerlaw = ((subj.beta) * subj.centwavs_meters**subj.alpha)
                #subj.nuFnu_disk_plotting_temp_powerlaw = np.zeros(plotting_xvec_angstroms.size)
                
                #print subj.centwavs_meters_to_use[-subj.num_excesses]*10000.
                
                powerlaw_plotting_component = np.array([((subj.beta) * ((x/(1.e10))**subj.alpha)) for x in plotting_xvec_angstroms if x >= (subj.centwavs_meters_to_use[-6]*1.e10)])
                powerlaw_plotting_component_length = powerlaw_plotting_component.size
                
                print powerlaw_plotting_component_length
                #print powerlaw_plotting_component
                
                subj.nuFnu_disk_plotting_temp_powerlaw = np.zeros(plotting_xvec_angstroms.size)
                subj.nuFnu_disk_plotting_temp_powerlaw[-powerlaw_plotting_component_length:] = powerlaw_plotting_component
                
                #print subj.nuFnu_disk_plotting_temp_powerlaw[-powerlaw_plotting_component_length:]
                
                subj.nuFnu_model_plotting_temp_powerlaw = np.array([max(subj.nuFnu_star_plotting_temp[i],subj.nuFnu_disk_plotting_temp_powerlaw[i]) for i in range(plotting_xvec_angstroms.size)])

            nuFnu_disk_points_blackbody = blackbody_lambda(subj.centwavs_meters * 1.e10 * u.AA, (10.**subj.log10Tdiskguess * u.K)) * np.pi * u.sr * (subj.centwavs_meters * 1.e10 * u.AA) * (10.**subj.log10xdiskguess) * u.cm * u.cm * u.s / u.erg
            subj.nuFnu_disk_plotting_temp_blackbody = blackbody_lambda(plotting_xvec_angstroms * u.AA, (10.**subj.log10Tdiskguess)*u.K) * np.pi * u.sr * (plotting_xvec_angstroms * u.AA) * (10.**subj.log10xdiskguess) * u.cm * u.cm * u.s / u.erg
            subj.nuFnu_model_plotting_temp_blackbody = subj.nuFnu_star_plotting_temp + subj.nuFnu_disk_plotting_temp_blackbody          

            if subj.powerlaw:
                subj.nuFnu_disk_plotting_temp = subj.nuFnu_disk_plotting_temp_powerlaw
                subj.nuFnu_model_plotting_temp = subj.nuFnu_model_plotting_temp_powerlaw                                                 
            else:
                subj.nuFnu_disk_plotting_temp = subj.nuFnu_disk_plotting_temp_blackbody
                subj.nuFnu_model_plotting_temp = subj.nuFnu_model_plotting_temp_blackbody
         
                #print nuFnu_disk_points
                #print subj.nuFnu_disk
                #nuFnu_disk_points = blackbody_lambda(cent_wavs_use*1.e10 *u.AA, subj.Tdisk * u)
        

                #print np.interp(subj.centwavs_meters*1.e10, plotting_xvec_angstroms, subj.nuFnu_disk_plotting)
        

                #subj.nuFnu_disk = subj.nuFnu - subj.nuFnu_star
    
                #subj.nuFnu_disk_errs = subj.nuFnuerrs
    
            return subj
    

subjs_with_disk_fits = []
subjs_with_powerlaw_fits = []

plt.close("all")

disk_fit_start_time = time.time()

subj_counter = 0

for subj in subjs_with_star_fits:
    print subj.zooniverse_id, subj.wiseid
    
    subj_counter += 1
    
    #if not subj.good_star_fit:
    #    print "No good star fit"
    #    subjs_with_disk_fits.append(subj)
    #else:
    print subj.num_excesses
    subjs_with_disk_fits.append(get_disk_fit(subj))
    if subj.num_excesses > 2:
        subjs_with_powerlaw_fits.append(subjs_with_disk_fits[-1])
    print subj.powerlaw
    print 'Tdisk:', 10.**(subj.log10Tdiskguess), 'xdisk', 10.**(subj.log10xdiskguess)
    print 'Alpha:', subj.alpha_guess, 'Beta:', 10.**(subj.log10beta_guess)
    #plot_subj_temp(subj)

    

    #if subj.num_excesses > 1:
    #    f1.write(subj.zooniverse_id+','+subj.wiseid+'\n')
    #elif subj.num_excesses < 1:
    #    f2.write(subj.zooniverse_id+','+subj.wiseid+'\n')
    #else:
    #    f3.write(subj.zooniverse_id+','+subj.wiseid+'\n')
    
    print len(subjs_with_disk_fits), subj_counter, (time.time() - disk_fit_start_time)/len(subjs_with_disk_fits)
    
disk_fit_end_time = time.time()

#f1.close()
#f2.close()
#f3.close()
    
print "Disk fits done"
print "Time elapsed:", disk_fit_end_time - disk_fit_start_time

def lnprior_models_powerlaw(theta, Teff):
    log10rdstar, alpha, log10beta = theta
    
    #Teff = 10.**log10Teff
    rdstar = 10.**log10rdstar
    #Tdisk = 10.**log10Tdisk
    #xdisk = 10.**log10xdisk
    
    #tefflogg = (Teff, logg)
    
    #print tefflogg
    
    #print keylist
    
    #Tefflogg_flag = False
    
    #for key in keylist:
    #    if (Teff - key[0] < 1.e-10) and (logg==key[1]):
    #        Tefflogg_flag = True
    #        break
            
    
    #Tefflogg_flag = (Teff, logg) in keylist
    
    rdstar_flag = rdstar < 1.22e-7
    
    alpha_flag = (alpha < 10.) and (alpha > -10.)
    beta_flag = log10beta < 0.
    
    #lir_lstar_flag = (((Tdisk/Teff)**4) * (xdisk/(rdstar**2))) < 1.
    
    #print 'Tefflogg', Tefflogg_flag
    #print 'rdstar', rdstar_flag
    #print 'Tdisk', Tdisk_flag
    #print 'lir/lstar', lir_lstar_flag
    
    if rdstar_flag and alpha_flag and beta_flag:
        return 0.0
    else:
        return -np.inf
    
def lnprior_blackbody_powerlaw(theta):
    log10Teff, log10rdstar, alpha, log10beta = theta
    
    #Teff = 10.**log10Teff
    #rdstar = 10.**log10rdstar
    #Tdisk = 10.**log10Tdisk
    #xdisk = 10.**log10xdisk
    
    #teff_flag = log10Teff < np.log10(30000.)
    #rdstar_flag = rdstar < 1.22e-7
    #Tdisk_flag = Tdisk < Teff
    #lir_lstar_flag = ((((Tdisk/Teff)**4) * (xdisk/(rdstar**2))) < 1.)
    
    if (log10Teff < 4.47712125472) and (log10rdstar < -6.91364016933) and (alpha < 10.) and (alpha > -10.) and (log10beta < -5.):
        return 0.0
    else:
        return -np.inf

    
def lnprob_models_powerlaw(theta, centwavs, nuFnu, nuFnuerrs, Teff, logg):
    lp = lnprior_models_powerlaw(theta, Teff)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_models(theta, centwavs, nuFnu, nuFnuerrs, Teff, logg)
    
    
def lnprob_blackbody_powerlaw(theta, centwavs, nuFnu, nuFnuerrs):
    lp = lnprior_blackbody_powerlaw(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_blackbody(theta, centwavs, nuFnu, nuFnuerrs)

print len(subjs_with_powerlaw_fits)

subjs_with_mcmc = []
mcmc_run_start_time = time.time()

print 'Starting MCMC', mcmc_run_start_time

f_final = open('disk_parameters_powerlaw.csv','w')

for subj in subjs_with_powerlaw_fits:
    
    print subj.zooniverse_id, subj.wiseid
    
    cent_wavs_full_use = subj.centwavs_meters_to_use
    nuFnu_full_use = subj.nuFnu_to_use
    nuFnu_full_err_use = subj.nuFnuerrs_to_use
    
    print subj.use_models
    if subj.use_models:
        ndim, nwalkers = 3, 100
        
        res = lnprob_models_powerlaw([subj.log10rdstarguess, subj.alpha, subj.log10beta], cent_wavs_full_use, nuFnu_full_use, nuFnu_full_err_use, subj.Teffguess, subj.logg_guess)
        pos = [[subj.log10rdstarguess, subj.alpha_guess, subj.log10beta_guess] +1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, 3, lnprob_models_powerlaw, args=(cent_wavs_full_use, nuFnu_full_use, nuFnu_full_err_use, subj.Teffguess, subj.logg_guess))
        
        sampler.run_mcmc(pos, 500)
		
		samples = sampler.chain[:, 150:, :].reshape((-1, ndim))
		
		subj.samples_use = np.zeros(samples.shape)
		print subj.samples_use.shape

		subj.samples_use[:,0] = 10.**samples[:,0]
		subj.samples_use[:,1] = samples[:,1]
		subj.samples_use[:,2] = 10.**samples[:,2]
		
		subj.rdstar = np.percentile(subj.samples_use[:,0],50)
		subj.rdstar_err_low = subj.rdstar - np.percentile(subj.samples_use[:,0], 16)
		subj.rdstar_err_high = np.percentile(subj.samples_use[:,0], 84) - subj.rdstar
		
		subj.alpha = np.percentile(subj.samples_use[:,1],50)
		subj.alpha_err_low = subj.alpha - np.percentile(subj.samples_use[:,1], 16)
		subj.alpha_err_high = np.percentile(subj.samples_use[:,1], 84) - subj.alpha

		subj.beta = np.percentile(subj.samples_use[:,2],50)
		subj.beta_err_low = subj.beta - np.percentile(subj.samples_use[:,2], 16)
		subj.beta_err_high = np.percentile(subj.samples_use[:,2], 84) - subj.beta

		subj.nuFnu_star_plotting = subj.nuFnu_star_plotting_temp

    else:
        ndim, nwalkers = 4, 100
        
        res = lnprob_blackbody_powerlaw([subj.log10Teffguess, subj.log10rdstarguess, subj.alpha, subj.log10beta], cent_wavs_full_use, nuFnu_full_use, nuFnu_full_err_use)
        pos = [[subj.log10Teffguess, subj.log10rdstarguess, subj.alpha, subj.log10beta] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, 4, lnprob_blackbody_powerlaw, args=(cent_wavs_full_use, nuFnu_full_use, nuFnu_full_err_use))
        
        sampler.run_mcmc(pos, 500)
    
		samples = sampler.chain[:, 150:, :].reshape((-1, ndim))
		print samples.shape
		
		subj.samples_use = np.zeros(samples.shape)
		print subj.samples_use.shape
		
		subj.samples_use[:,0] = 10.**samples[:,0]
		subj.samples_use[:,1] = 10.**samples[:,1]
		subj.samples_use[:,2] = samples[:,2]
		subj.samples_use[:,3] = 10.**samples[:,3]

		subj.Teff = np.percentile(subj.samples_use[:,0],50)
		subj.Teff_err_low = subj.Teff - np.percentile(subj.samples_use[:,0], 16)
		subj.Teff_err_high = np.percentile(subj.samples_use[:,0], 84) - subj.Teff
		
		subj.rdstar = np.percentile(subj.samples_use[:,1],50)
		subj.rdstar_err_low = subj.rdstar - np.percentile(subj.samples_use[:,1], 16)
		subj.rdstar_err_high = np.percentile(subj.samples_use[:,1], 84) - subj.rdstar
		
		subj.alpha = np.percentile(subj.samples_use[:,2],50)
		subj.alpha_err_low = subj.alpha - np.percentile(subj.samples_use[:,2], 16)
		subj.alpha_err_high = np.percentile(subj.samples_use[:,2], 84) - subj.alpha

		subj.beta = np.percentile(subj.samples_use[:,3],50)
		subj.beta_err_low = subj.beta - np.percentile(subj.samples_use[:,3], 16)
		subj.beta_err_high = np.percentile(subj.samples_use[:,3], 84) - subj.beta

		subj.nuFnu_star_plotting = blackbody_lambda(plotting_xvec_angstroms * u.AA, subj.Teff*u.K) * np.pi * u.sr * (plotting_xvec_angstroms * u.AA) * (subj.rdstar**2) * u.cm * u.cm * u.s / u.erg

			

    subjs_with_mcmc.append(subj)
    print str(subj)
	f_final.write(str(subj) + '\n')
	
	
    obj_fin_time = time.time()
        
    print obj_fin_time, len(subjs_with_mcmc), (obj_fin_time - mcmc_run_start_time)/float(len(subjs_with_mcmc))
    print ''
	
f_final.close()

print 'MCMC finished. Runtime:', obj_fin_time - mcmc_run_start_time