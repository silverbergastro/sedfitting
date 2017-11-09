import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import *
import os.path
from scipy.constants import h,k,c
from scipy.optimize import curve_fit, brute
import scipy as sy
import pandas
from astropy.io import ascii
import time
#from synth_phot import fluxdrive_star, fluxdrive_plot, fluxdrive_disk, fluxdrive_disk_test, import_filter, uniq

def import_filter(filtername):
    '''Import filter information--response curve'''
    filterfilename = filtername+"_response.dat" #Filename

    #Initialize lists
    filter_xlist = []
    filter_Slist = []

    #Initialize dict
    filter_dict = {}

    #Read in file
    with open(filterfilename) as f:
        filterfilelist = f.readlines()

    #testline = filterfilelist[0]

    #translate file into two lists and a dict
    for line in filterfilelist:
        datavec = line.split()
        x = float(datavec[0])
        s = float(datavec[1])
        filter_xlist.append(x)
	filter_Slist.append(s)
        filter_dict[x] = s

    #convert lists to numpy array
    filter_X = np.array(filter_xlist)
    filter_S = np.array(filter_Slist)

    return filter_X, filter_S, filter_dict
    
def uniq(inputlist):
    '''Get unique values from two different wavelength arrays'''

    seen = set() #Initialize set
    seen_add = seen.add #Initialize other function
    return [x for x in inputlist if not (x in seen or seen_add(x))]

def synth_phot(filt_dict, spec_X, f_lambda_slightly_less_wrong_units, den):
    filt_X = np.array(sorted(filt_dict.keys()))
    filt_S = np.zeros(filt_X.size)
    #Make sure wavelength/sensitivity correspondence works
    for i in range(filt_S.size):
        filt_S[i] = filt_dict[filt_X[i]]

    #Convert filtX to list
    temp_filtx = list(filt_X)

    #Get limits for filter size
    wmin = min(filt_X)
    wmax = max(filt_X)

    #Get wavelength range inside filter
    temp_specx1 = [i for i in list(spec_X) if i >= wmin]
    temp_specx2 = [i for i in temp_specx1 if i <= wmax]

    #Create grid of filtx and specx, in order, for convolution purposes
    wgridlist = temp_filtx + temp_specx2
    wgridlist.sort()

    #Reduce grid to unique values
    wgrid_use = uniq(wgridlist)
    wgrid = np.array(wgrid_use)

    #Interpolate filter sensitivity onto combined grid
    Sg = np.interp(wgrid, filt_X, filt_S)

    #Interpolate f_lambda onto combined grid
    Sf = np.interp(wgrid, spec_X, f_lambda_slightly_less_wrong_units) 

    #Multiply response curve by f_lambda, integrate over
    num = np.trapz(Sf*Sg, x=wgrid)

    return num/den
	
#Rename variable for clarity
kB = k

#Import all filters
g_X, g_S, g_dict = import_filter('g')
r_X, r_S, r_dict = import_filter('r')
i_X, i_S, i_dict = import_filter('i')
z_X, z_S, z_dict = import_filter('z')
y_X, y_S, y_dict = import_filter('y')
J_X, J_S, J_dict = import_filter('J')
H_X, H_S, H_dict = import_filter('H')
K_X, K_S, K_dict = import_filter('K')
W1_X, W1_S, W1_dict = import_filter('W1')
W2_X, W2_S, W2_dict = import_filter('W2')
W3_X, W3_S, W3_dict = import_filter('W3')
W4_X, W4_S, W4_dict = import_filter('W4')

#Create dictionary of filter dictionaries
filter_dict = {}
filter_dict['g'] = g_dict
filter_dict['r'] = r_dict
filter_dict['i'] = i_dict
filter_dict['z'] = z_dict
filter_dict['y'] = y_dict
filter_dict['J'] = J_dict
filter_dict['H'] = H_dict
filter_dict['K'] = K_dict
filter_dict['W1'] = W1_dict
filter_dict['W2'] = W2_dict
filter_dict['W3'] = W3_dict
filter_dict['W4'] = W4_dict

#Create dictionary of denominators for flux calculation
dens_dict = {}

#Initialize wavelength array for getting denominators
spec_X_filterbuild = np.linspace(0.5, 50., 49501)

#Initialize lists of filter names
has_ps_filter_list = ['g','r','i','z','y','J','H','K','W1','W2','W3','W4']
no_ps_filter_list = ['J','H','K','W1','W2','W3','W4']

for filt in filter_dict.keys():
    filt_dict = filter_dict[filt]
    filt_X = np.array(sorted(filt_dict.keys()))
    filt_S = np.zeros(filt_X.size)
    #Make sure wavelength/sensitivity correspondence works
    for i in range(filt_S.size):
        filt_S[i] = filt_dict[filt_X[i]]

    #Convert filtX to list
    temp_filtx = list(filt_X)

    #Get limits for filter size
    wmin = min(filt_X)
    wmax = max(filt_X)

    #Get wavelength range inside filter
    temp_specx1 = [i for i in list(spec_X_filterbuild) if i >= wmin]
    temp_specx2 = [i for i in temp_specx1 if i <= wmax]

    #Create grid of filtx and specx, in order, for convolution purposes
    wgridlist = temp_filtx + temp_specx2
    wgridlist.sort()

    wgrid_use = uniq(wgridlist)
    wgrid = np.array(wgrid_use)

    #Interpolate filter sensitivity onto combined grid
    Sg = np.interp(wgrid, filt_X, filt_S)

    #Integrate over the filter curve
    den = np.trapz(Sg, x=wgrid)
    
    #Store integrated value in array
    dens_dict[filt] = den

#Initialize dictionaries of filter characteristics
central_wavelengths_dict = {}
zero_points_dict = {}

#Read in filter characteristics
data = ascii.read("filter_characteristics.txt")

for i in range(12):
    #Store central wavelengths and zero points for all filters in dictionaries
    central_wavelengths_dict[data['filt'][i]] = data['centwav'][i]
    zero_points_dict[data['filt'][i]] = data['zp'][i]

def fcn2min_getval(x_wav_data999, Teff, frac_r_d):
    '''Function to minimize to get flux data'''
    #Teff = params[0]
    #frac_r_d = params[1]
    #has_ps = False
    
    #Get filter names
    #if has_ps:
    #    filter_list_use = has_ps_filter_list[:9]
    #else:
    #    filter_list_use = no_ps_filter_list[:4]

    filter_list_use = no_ps_filter_list

    #print len(filter_list_use)

    #Get wavelength space to integrate over
    spec_X = np.linspace(0.5, 50., 49501)   #In microns
    spec_X_met = spec_X*1.e-6               #And in meters

    
    tempBB_star = planck_lambda(spec_X_met, Teff)   #Get blackbody function for this temperature
    factor_star = frac_r_d

    f_lambda_wrong_units = np.pi * tempBB_star         #Put flux in units of Wm**-3
    f_lambda_slightly_less_wrong_units = f_lambda_wrong_units * 1.e-6  #Put flux in units of W/m^2/um
    
    c_microns = c*1.e6 #Get speed of light in microns/second
    
    #for i in range(spec_X.size):
    #    f_lambda_right_units.append(f_lambda_disk_slightly_less_wrong_units[i] * spec_X[i] * spec_X[i] / c_microns) #Convert to units of W/m^2/Hz
   
    f_nu_slightly_less_wrong_units = []
 
    for filter in filter_list_use:
        #Get filter-weighted flux in each band in W/m^2/um
        f_nu_slightly_less_wrong_units.append(synth_phot(filter_dict[filter], spec_X, f_lambda_slightly_less_wrong_units, dens_dict[filter]))

    f_nu_slightly_less_wrong_units_dist_corrected = np.array(f_nu_slightly_less_wrong_units)*factor_star #correct for stellar radius, distance to star
    
    nu_fnu_right_units_dist_corrected = []
    
    for i in range(len(filter_list_use)):
        nu_fnu_right_units_dist_corrected = central_wavelengths_dict[filter_list_use[i]] * f_nu_slightly_less_wrong_units_dist_corrected
    
    return nu_fnu_right_units_dist_corrected
    
def fcn2min_hasps_getval(x_wav_data999, Teff, frac_r_d):
    '''Function to minimize to get flux data'''
    #Teff = params[0]
    #frac_r_d = params[1]
    #has_ps = False
    
    #Get filter names
    #if has_ps:
    #    filter_list_use = has_ps_filter_list[:9]
    #else:
    #    filter_list_use = no_ps_filter_list[:4]

    filter_list_use = no_ps_filter_list

    #print len(filter_list_use)

    #Get wavelength space to integrate over
    spec_X = np.linspace(0.5, 50., 49501)   #In microns
    spec_X_met = spec_X*1.e-6               #And in meters

    
    tempBB_star = planck_lambda(spec_X_met, Teff)   #Get blackbody function for this temperature
    factor_star = frac_r_d

    f_lambda_wrong_units = np.pi * tempBB_star         #Put flux in units of Wm**-3
    f_lambda_slightly_less_wrong_units = f_lambda_wrong_units * 1.e-6  #Put flux in units of W/m^2/um
    
    c_microns = c*1.e6 #Get speed of light in microns/second
    
    #for i in range(spec_X.size):
    #    f_lambda_right_units.append(f_lambda_disk_slightly_less_wrong_units[i] * spec_X[i] * spec_X[i] / c_microns) #Convert to units of W/m^2/Hz
   
    f_nu_slightly_less_wrong_units = []
 
    for filter in filter_list_use:
        #Get filter-weighted flux in each band in W/m^2/um
        f_nu_slightly_less_wrong_units.append(synth_phot(filter_dict[filter], spec_X, f_lambda_slightly_less_wrong_units, dens_dict[filter]))

    f_nu_slightly_less_wrong_units_dist_corrected = np.array(f_nu_slightly_less_wrong_units)*factor_star #correct for stellar radius, distance to star
    
    nu_fnu_right_units_dist_corrected = []
    
    for i in range(len(filter_list_use)):
        nu_fnu_right_units_dist_corrected = central_wavelengths_dict[filter_list_use[i]] * f_nu_slightly_less_wrong_units_dist_corrected
    
    return nu_fnu_right_units_dist_corrected
    
def fcn2min(x_wav_data999, Teff, frac_r_d):
    '''Function to minimize to get flux data'''
    #Teff = params[0]
    #frac_r_d = params[1]
    #has_ps = False
    
    #Get filter names
    #if has_ps:
    #    filter_list_use = has_ps_filter_list[:9]
    #else:
    #    filter_list_use = no_ps_filter_list[:4]

    filter_list_use = no_ps_filter_list[:4]

    #print len(filter_list_use)

    #Get wavelength space to integrate over
    spec_X = np.linspace(0.5, 50., 49501)   #In microns
    spec_X_met = spec_X*1.e-6               #And in meters

    
    tempBB_star = planck_lambda(spec_X_met, Teff)   #Get blackbody function for this temperature
    factor_star = frac_r_d

    f_lambda_wrong_units = np.pi * tempBB_star         #Put flux in units of Wm**-3
    f_lambda_slightly_less_wrong_units = f_lambda_wrong_units * 1.e-6  #Put flux in units of W/m^2/um
    
    c_microns = c*1.e6 #Get speed of light in microns/second
    
    #for i in range(spec_X.size):
    #    f_lambda_right_units.append(f_lambda_disk_slightly_less_wrong_units[i] * spec_X[i] * spec_X[i] / c_microns) #Convert to units of W/m^2/Hz
   
    f_nu_slightly_less_wrong_units = []
 
    for filter in filter_list_use:
        #Get filter-weighted flux in each band in W/m^2/um
        f_nu_slightly_less_wrong_units.append(synth_phot(filter_dict[filter], spec_X, f_lambda_slightly_less_wrong_units, dens_dict[filter]))

    f_nu_slightly_less_wrong_units_dist_corrected = np.array(f_nu_slightly_less_wrong_units)*factor_star #correct for stellar radius, distance to star
    
    nu_fnu_right_units_dist_corrected = []
    
    for i in range(len(filter_list_use)):
        nu_fnu_right_units_dist_corrected = central_wavelengths_dict[filter_list_use[i]] * f_nu_slightly_less_wrong_units_dist_corrected
    
    return nu_fnu_right_units_dist_corrected
    
def fcn2min_hasps(x_wav_data999, Teff, frac_r_d):
    '''Function to minimize to get flux data'''
    #Teff = params[0]
    #frac_r_d = params[1]
    #has_ps = False
    
    #Get filter names
    #if has_ps:
    #    filter_list_use = has_ps_filter_list[:9]
    #else:
    #    filter_list_use = no_ps_filter_list[:4]

    filter_list_use = no_ps_filter_list[:9]

    #print len(filter_list_use)

    #Get wavelength space to integrate over
    spec_X = np.linspace(0.5, 50., 49501)   #In microns
    spec_X_met = spec_X*1.e-6               #And in meters

    
    tempBB_star = planck_lambda(spec_X_met, Teff)   #Get blackbody function for this temperature
    factor_star = frac_r_d

    f_lambda_wrong_units = np.pi * tempBB_star         #Put flux in units of Wm**-3
    f_lambda_slightly_less_wrong_units = f_lambda_wrong_units * 1.e-6  #Put flux in units of W/m^2/um
    
    c_microns = c*1.e6 #Get speed of light in microns/second
    
    #for i in range(spec_X.size):
    #    f_lambda_right_units.append(f_lambda_disk_slightly_less_wrong_units[i] * spec_X[i] * spec_X[i] / c_microns) #Convert to units of W/m^2/Hz
   
    f_nu_slightly_less_wrong_units = []
 
    for filter in filter_list_use:
        #Get filter-weighted flux in each band in W/m^2/um
        f_nu_slightly_less_wrong_units.append(synth_phot(filter_dict[filter], spec_X, f_lambda_slightly_less_wrong_units, dens_dict[filter]))

    f_nu_slightly_less_wrong_units_dist_corrected = np.array(f_nu_slightly_less_wrong_units)*factor_star #correct for stellar radius, distance to star
    
    nu_fnu_right_units_dist_corrected = []
    
    for i in range(len(filter_list_use)):
        nu_fnu_right_units_dist_corrected = central_wavelengths_dict[filter_list_use[i]] * f_nu_slightly_less_wrong_units_dist_corrected
    
    return nu_fnu_right_units_dist_corrected
    
def fcn2min1(x_wav_data999, *params):
    '''Function to minimize to get flux data'''
    Teff = params[0]
    xfactor = params[1]
    has_ps = False
    
    if x_wav_data999[0] < 1.:
        has_ps = True
        
    #Get filter names
    if has_ps:
        filter_list_use = has_ps_filter_list
    else:
        filter_list_use = no_ps_filter_list

    #Get wavelength space to integrate over
    spec_X = np.linspace(0.5, 50., 49501)   #In microns
    spec_X_met = spec_X*1.e-6               #And in meters

    
    tempBB_disk = planck_lambda(spec_X_met, Teff)   #Get blackbody function for this temperature in W m^-3 sr^-1
    factor_disk = xfactor

    f_lambda_wrong_units = np.pi * tempBB_disk         #Put flux in units of Wm**-3
    f_lambda_slightly_less_wrong_units = f_lambda_wrong_units * 1.e-6  #Put flux in units of W/m^2/um
    
    c_microns = c*1.e6 #Get speed of light in microns/second
    
    #for i in range(spec_X.size):
    #    f_lambda_right_units.append(f_lambda_disk_slightly_less_wrong_units[i] * spec_X[i] * spec_X[i] / c_microns) #Convert to units of W/m^2/Hz
   
    f_nu_slightly_less_wrong_units = []
 
    for filter in filter_list_use:
        #Get filter-weighted flux in each band in W/m^2/um
        f_nu_slightly_less_wrong_units.append(synth_phot(filter_dict[filter], spec_X, f_lambda_slightly_less_wrong_units, dens_dict[filter]))

    f_nu_slightly_less_wrong_units_dist_corrected = np.array(f_nu_slightly_less_wrong_units)*xfactor #correct for stellar radius, distance to star
    
    nu_fnu_right_units_dist_corrected = []
    
    for i in range(len(filter_list_use)):
        nu_fnu_right_units_dist_corrected = central_wavelengths_dict[filter_list_use[i]] * f_nu_slightly_less_wrong_units_dist_corrected
    
    return nu_fnu_right_units_dist_corrected

def chi2min(params, *arguments):
    xvec, yvec, sigvec = arguments
    #print "params", params
    synth_fluxes = fcn2min1(xvec, *params)
    dif = synth_fluxes - yvec
    
    chisq = np.sum((dif/sigvec)**2)
    return chisq
    
    
def planck_lambda(wavelength, temperature):
    '''Calculate Planck function for given vector of wavelengths and scalar temperature. Can be replaced by astrolibR Planck function, but unit conversion will be needed
    and multiplication by pi in above function will be covered by built-in.'''
    resvec = np.zeros(wavelength.size)

    for i in range(wavelength.size):
        frontfrac = (2.*h*c*c)/(wavelength[i]**5)

	#print 'h', h
	#print 'c', c
	#print 'lambda', wavelength[i]
	#print 'k', k
	#print 'T', temperature
	#print 'h*c', h*c
	#print 'lambda*k*T', wavelength[i]*k*temperature
	#print (h*c)/(wavelength[i]*k*temperature)
	exp_part = np.exp((h*c)/(wavelength[i]*kB*temperature))
        #print exp_part
        backfrac = 1./(exp_part - 1.)

	#print frontfrac, backfrac

        resvec[i] = frontfrac*backfrac

    return resvec

def planck_nu(frequency, temperature):

    resvec = np.zeros(frequency.size)

    for i in range(frequency.size):
	frontfrac = (2.*h*(frequency[i]**3))/(c**2)
	backfrac = 1./(exp((h*frequency[i])/(k*temperature)) - 1.)
	resvec[i] = frontfrac*backfrac

    return resvec


def count_excess_bands(difs, errors):

    #difs = np.array(real_fluxes) - np.array(synth-fluxes)

    sigmas = np.array(difs)/np.array(errors)

    num_sig_excesses = 0
    for i in range(sigmas.size):
	if sigmas[i] > 5:
	    num_sig_excesses += 1

    return num_sig_excesses
    
    
def driver(filename):
    '''Driver function for getting fit and plot for single object. Calls function getdata to get data in a convenient vector form. Translates data into usable format. Calls fitting 
    function. Calls plotting function.'''
    
    '''Vector includes WISE ID, and alternating magnitudes and errors, in order: Jmag, Jerr, Hmag, Herr, Kmag Kerr, W1mag, W1err, W2mag, W2err, W3mag, W3err, W4mag, W4err, gmag, gerr, 
    rmag, rerr, imag, ierr, zmag, zerr, ymag, yerr. Convert magnitudes to fluxes, calls fitting function to get best fit stellar temperature and radius/distance factor (r^2/d^2).'''
    
    df = pandas.read_csv(filename)
    data = df.values
    
    f1 = open('characteristic_outputs.csv','w')
    
    timer_start = -1.
    
    time_list = []
    last_time = []

    counter = 0

    for line in data:
        timer = time.time()
        has_ps = False
        nancheck = False
        has_val = True
        write_string = ''
    
        #Read datavec into variable
        wise_id = line[0]
        w1mag = line[1]
        w1err = line[2]
        w2mag = line[3]
        w2err = line[4]
        w3mag = line[5]
        w3err = line[6]
        w4mag = line[7]
        w4err = line[8]
        jmag = line[9]
        jerr = line[10]
        hmag = line[11]
        herr = line[12]
        kmag = line[13]
        kerr = line[14]
        gmag = line[15]
        gerr = line[16]
        imag = line[17]
        ierr = line[18]
        rmag = line[19]
        rerr = line[20]
        ymag = line[21]
        yerr = line[22]
        zmag = line[23]
        zerr = line[24]

        #for i in range(15,25):
        #    if np.isnan(line[i]) and not nancheck:
        #        nancheck = True

        #for i in range(15,25):
        #    if data[i] < -998. and not has_val:
        #        has_val = True
                
        #if not nancheck and not has_val:
        #    has_ps = True
        
        
        #List of filter names in both cases
        if has_ps:
            filt_names = ['g','r','i','z','y','J','H','K','W1','W2','W3','W4']
        else:
            filt_names = ['J','H','K','W1','W2','W3','W4']
            
        w1_w4_diff = w1mag - w4mag
        w1_w4_unc = np.sqrt((w1err**2)+(w4err**2))
        
        w1_w3_diff = w1mag - w3mag
        w1_w3_unc = np.sqrt((w1err**2)+(w3mag**2))
        
        mag_excess_difs = (w1_w4_diff, w1_w4_unc, w1_w3_diff, w1_w3_unc)
        
        filt_cent_wav = []
        filt_zp = []
        mag_vec = []
        mag_err_vec = []
        flux_vec = []
        flux_err_vec = []
        
        
        for filt_name in filt_names:
            filt_cent_wav.append(central_wavelengths_dict[filt_name])
            filt_zp.append(zero_points_dict[filt_name])
        
        
        if has_ps:
            #put mags and errors into vectors
            mag_vec = [gmag, rmag, imag, zmag, ymag, jmag, hmag, kmag, w1mag, w2mag, w3mag, w4mag]
            mag_err_vec = [gerr, rerr, ierr, zerr, yerr, jerr, herr, kerr, w1err, w2err, w3err, w4err]
        else:
            mag_vec = [jmag, hmag, kmag, w1mag, w2mag, w3mag, w4mag]
            mag_err_vec = [jerr, herr, kerr, w1err, w2err, w3err, w4err]
            
        for i in range(len(filt_names)):
            #convert mags to fluxes
            flux_vec.append(get_flux(filt_zp[i], mag_vec[i]))
            #Convert errors
            flux_err_vec.append(get_flux_err(filt_zp[i], mag_vec[i], mag_err_vec[i]))
           
        #print filt_cent_wav
        #print flux_vec
        #print flux_err_vec
        print wise_id
   
        #Get fits
        Tstar, Tstar_error, rdstar, rdstar_error, Tdisk, Tdisk_error, xdisk, xdisk_error, lir_lstar, lir_lstar_error = get_fits(filt_cent_wav, np.array(flux_vec), np.array(flux_err_vec), has_ps)
        
        
        write_string = wise_id+','+str(Tstar)+','+str(Tstar_error)+','+str(rdstar)+','+str(rdstar_error)+','+str(Tdisk)+','+str(Tdisk_error)+','+str(xdisk)+','+str(xdisk_error)+','+str(lir_lstar)+','+str(lir_lstar_error) +'\n'
        
        f1.write(write_string)

        print time.time() - timer
        
        #Plotting
        #plot_status = plot_sed(wise_id, Tstar, Tstar_error, rdstar, rdstar_error, Tdisk, Tdisk_error, xdisk, xdisk_error, lir_lstar, lir_lstar_error, mag_excess_difs)
    
    print "Done"
    
            
def get_flux(zero_point, magnitude):
    '''Get flux in W m^-2'''
    return zero_point*(10.**(-0.4*magnitude)) * 1.e-26
    
def get_flux_err(zero_point, magnitude, mag_err):
    '''Get flux error in W m^-2'''
    flux_jansky = get_flux(zero_point, magnitude) * 1.e26
    deriv = -0.4*flux_jansky*np.log(10.)
    
    err_jansky = np.sqrt(deriv**2 * mag_err**2)
    
    return err_jansky*1.e-26


def get_fits(filt_cent_wav, flux_vec, flux_err_vec, has_ps):
    '''Get fits for star and disk. Takes wavelength, flux, error, has_ps from driver, returns fits for stellar temperature, distance factor rdstar^2, disk temperature, distance factor
    xdisk, and associated errors. Also return lir/lstar, the fractional infrared luminosity of the disk.'''
    
    #Set initial guess for stellar parameters--solar analogue 50 pc away
    params_star = sy.array([5775., 2.e-19])

    #if has_ps:
    #    filt_cent_wav_use = 
    
    #Get fit for star, using only W1 and bluer filters
    if has_ps:
        popt_star, pcov_star = curve_fit(fcn2min_hasps, np.array(filt_cent_wav[:-3]), np.array(flux_vec[:-3]), p0=params_star, sigma=np.array(flux_err_vec[:-3]), absolute_sigma=True)
    else:
        popt_star, pcov_star = curve_fit(fcn2min, np.array(filt_cent_wav[:-3]), np.array(flux_vec[:-3]), p0=params_star, sigma=np.array(flux_err_vec[:-3]), absolute_sigma=True)
        
    #Use optimum fit parameters to get synthetic fluxes for all bands
    if has_ps:
        synth_fluxes = fcn2min_hasps_getval(np.array(filt_cent_wav), *popt_star)
    else:
        synth_fluxes = fcn2min_getval(np.array(filt_cent_wav), *popt_star)
        
    Tstar_final = popt_star[0]
    rdstar_final = popt_star[1]
    star_errors = np.sqrt(np.diag(pcov_star))
    Tstar_error = star_errors[0]
    rdstar_error = star_errors[1]
    #Subtract synthesized fluxes from actual fluxes to get remainder/excess
    remain_fluxes = np.array(flux_vec) - np.array(synth_fluxes)
    
    #Determine number of bands in excess--will dictate how disk fit is done
    num_sig_excesses = count_excess_bands(remain_fluxes, flux_err_vec)
    print num_sig_excesses   
 
    if num_sig_excesses > 1:
        #Two excesses--do normal least-squares minimization
        
        #Set initial guesses: Tdisk = 150., xdisk = 1.e-4*((Teff/Tdisk)**4)*rd
        params_disk = [150., 1.e-4*rdstar_final*((Tstar_final/150.)**4)]
        
        #Get parameters
        popt_disk, pcov_disk = curve_fit(fcn2min1, filt_cent_wav, remain_fluxes, p0=params_disk, sigma=flux_err_vec, absolute_sigma=True)
        
        Tdisk_final = popt_disk[0]
        xdisk_final = popt_disk[1]
        
        disk_errors = np.sqrt(np.diag(pcov_disk))
        Tdisk_error = disk_errors[0]
        xdisk_error = disk_errors[1]
        
    elif num_sig_excesses == 0:
        #No significant excess--don't need to fit
        print "No excesses detected."
        
        Tdisk_final = 0.
        xdisk_final = 0.
        
        Tdisk_error = 0.
        xdisk_error = 0.
        
    else:
        #One point of excess--way more complicated.
        
        #Set up search parameter ranges for brute force disk fit
        #ranges_Tdisk = slice(0., Teff_final, 100.)
        #ranges_xdisk = slice(0., rdstar_final, (rdstar_final/10.))
        rranges = (slice(0., Tstar_final, 200.), slice(0., rdstar_final, (rdstar_final/10.)))
        
        #Get approximate parameters by brute force
        resbrute = brute(chi2min, rranges, args=(np.array(filt_cent_wav), np.array(remain_fluxes), np.array(flux_err_vec)), full_output=True)
        
        #Use approximate parameters as guess
        params_disk = resbrute[0]
        
        #Get accurate parameters
        popt_disk, pcov_disk = curve_fit(fcn2min1, np.array(filt_cent_wav), np.array(remain_fluxes), p0=params_disk, sigma=np.array(flux_err_vec), absolute_sigma=True)
        
        Tdisk_final = popt_disk[0]
        xdisk_final = popt_disk[1]
        
        disk_errors = np.sqrt(np.diag(pcov_disk))
        Tdisk_error = disk_errors[0]
        xdisk_error = disk_errors[1]
        
    #Compute approximate fractional infrared luminosity
    lir_lstar = ((Tdisk_final/Tstar_final)**4)*(xdisk_final/rdstar_final)
    
    #Compute error in fractional infrared luminosity
    
    #Compute derivatives of lir_lstar with respect to each variable
    dlir_dTdisk = 4.*((Tdisk_final/Tstar_final)**3)*(xdisk_final/rdstar_final)*(1./Tstar_final)
    dlir_dTstar = 4.*((Tdisk_final/Tstar_final)**3)*(xdisk_final/rdstar_final)*(-Tdisk_final/(Tstar_final**2))
    dlir_dx = ((Tdisk_final/Tstar_final)**4) * (xdisk_final/rdstar_final)
    dlir_drd = ((Tdisk_final/Tstar_final)**4) * (-xdisk_final/(rdstar_final**2))
    
    
    #Compute error through standard error propagation
    lir_lstar_error = np.sqrt(((dlir_dTdisk**2)*(Tdisk_error**2)) + ((dlir_dTstar**2)*(Tstar_error**2)) + ((dlir_dx**2)*(xdisk_error**2)) + ((dlir_drd**2)*(rdstar_error**2)))
    
    return Tstar_final, Tstar_error, rdstar_final, rdstar_error, Tdisk_final, Tdisk_error, xdisk_final, xdisk_error, lir_lstar, lir_lstar_error
    
def plot_sed(designation, Tstar, Tstar_error, rdstar, rdstar_error, Tdisk, Tdisk_error, xdisk, xdisk_error, lir_lstar, lir_lstar_error,filt_cent_wav, flux_vec, flux_err_vec, mag_excess_difs):
    '''Takes input parameters and actually plots them. Loglog plots of nu*fnu (lambda*flambda) in units of (erg s^-1 cm^-2) versus wavelength (um).'''
    
    #Set plotting limits
    #ideally these aren't necessary and we can plot longer wavelengths as necessary just by panning right, but that probably takes up a non-negligible amount of storage space
    min_wav = 0.3
    max_wav = 100.
    
    logminwav = np.log10(min_wav)
    logmaxwav = np.log10(max_wav)
    
    #Generate log-wavelength vector, approximately spaced in increments of 0.00001)
    logwav = np.linspace(logminwav, logmaxwav, 252289)
    
    #Generate wavelength-space version of logwav--units are microns
    wav = 10**logwav
    
    wav_meters = 1.e-6*wav
    
    #Get f_lambda of star in units of W m^-3
    f_lambda_star_wrong_units = np.pi*np.array(planck_lambda(wav_meters, Tstar))*rdstar
    
    #Convert f_lambda to erg s^-1 cm^-2 um^-1
    f_lambda_star = 1.e-3*f_lambda_star_wrong_units
    
    #Get lamb_f_lambda_star
    lamb_f_lamb_star = f_lambda_star*wav
    
    #Do same stuff for disk
    f_lambda_disk = 1.e-3*np.pi*np.array(planck_lambda(wav_meters, Tstar))*rdstar
    lamb_f_lamb_disk = f_lambda_disk*wav
    
    total_model = lamb_f_lamb_star + lamb_f_lamb_disk
    
    #Generature figure
    plt.figure()
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
    
    lamb_flux = np.array(filt_cent_wav) * np.array(flux_vec)
    lamb_flux_error = np.array(filt_cent_wav) * np.array(flux_err_vec)

    ax = plt.gca()
    #plt.gcf.subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(right=0.92)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(top=0.95) 
    p16, = ax.plot(wav,lamb_f_lamb_star,'b--',linewidth=3.,label="Star Blackbody Model")
    p36, = ax.plot(wav,lamb_f_lamb_disk,'g',dashes=[8, 4, 2, 4, 2, 4],linewidth=3.,label="Disk Blackbody Model")
    p26, = ax.plot(wav,total_model,'black',linewidth=3.5,label="Total Blackbody Model")
    ax.errorbar(filt_cent_wav,lamb_flux,yerr=lamb_flux_err, fmt='p',color = 'red', markersize=14)
    ax.set_xlabel(r'$\mathrm{Wavelength(\mu m)}$',fontsize=24)
    ax.set_ylabel(r'$\mathrm{\nu F_\nu(erg\,s^{-1}\,cm^{-2})}$',fontsize=24)
    #time.sleep(4)
    ax.set_yscale('log')
    ax.set_xscale('log')

    #ax.text(0.075, 0.925, 'b)', va='top', color = 'black', transform=ax.transAxes, fontsize=20)

    ax.set_ylim([1.e-15, 1.e-9])

    w1_w4_diff, w1_w4_unc, w1_w3_diff, w1_w3_unc = mag_excess_diffs
    
    print designation
    print "T_star =", Tstar, "p/m", T_star_error
    print "T_disk =", Tdisk, "p/m", Tdisk_error
    print "f =", lir_lstar, "p/m", lir_lstar_error
    print "W1 - W3 =", w1_w3_diff, "p/m", w1_w3_unc
    print "W1 - W4 =", w1_w4_diff, "p/m", w1_w4_unc

    plt.legend(handles=[p16,p26,p36], loc='upper right',borderpad=.5, labelspacing=.2)
        
    path = os.path.join(designation)
#        print (path)
#        plt.show()
    frmt = '.png'
    total_path = path + frmt
#        print (total_path)
    plt.savefig(total_path, fmt='.png', dpi=1000)
#        plt.show()
    
    done_status = "Done"
    
    return done_status
