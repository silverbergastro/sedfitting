import numpy as np
import math
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os.path
from scipy.constants import h,k,c
from scipy import *
from scipy import optimize
from lmfit import minimize, Parameters, Parameter, report_fit
import time
from find_errors import find_errors
from temp_shambo_fit_star import temp_shambo_fit_star
from temp_shambo_fit_disk import temp_shambo_fit_disk
from synth_phot import fluxdrive_star, fluxdrive_plot, fluxdrive_disk, fluxdrive_disk_test, import_filter, uniq
from find_errors_spect import find_errors_spect

from matplotlib.patches import Rectangle
count=0

mag_table_file_2MASS_WISE = 'colmag.BT-Settl.server.2MASS_WISE.Vega'
mag_dict_2MASS_WISE = {}

mag_table_file_DENIS = 'BT_Settl_DENIS_mags'
#mag_table_file_DENIS = 'colmag.BT-Settl.server.DENIS.Vega'
mag_dict_DENIS = {}

mag_table_file_Tycho2 = 'BT_Settl_Tycho2_mags'
mag_dict_Tycho2 = {}

mag_table_file_APASS = 'BT_Settl_APASS_mags.txt'
mag_dict_APASS = {}

#print h, k, c

kB = k

K_X, K_S, K_dict = import_filter('K')
W1_X, W1_S, W1_dict = import_filter('W1')
W2_X, W2_S, W2_dict = import_filter('W2')
W3_X, W3_S, W3_dict = import_filter('W3')
W4_X, W4_S, W4_dict = import_filter('W4')

filter_dict = {}

filter_dict['K'] = K_dict
filter_dict['W1'] = W1_dict
filter_dict['W2'] = W2_dict
filter_dict['W3'] = W3_dict
filter_dict['W4'] = W4_dict

dens_dict = {}

spec_X_filterbuild = np.linspace(0.5, 50., 49501)

for filt in filter_dict.keys():
    filt_dict = filter_dict[filt]
    
    filt_X = np.array(sorted(filt_dict.keys()))

    filt_S = np.zeros(filt_X.size)

    for i in range(filt_S.size):
        filt_S[i] = filt_dict[filt_X[i]]

    temp_filtx = list(filt_X)

    wmin = min(filt_X)
    wmax = max(filt_X)

    temp_specx1 = [i for i in list(spec_X_filterbuild) if i >= wmin]
    temp_specx2 = [i for i in temp_specx1 if i <= wmax]

    wgridlist = temp_filtx + temp_specx2
    wgridlist.sort()

    wgrid_use = uniq(wgridlist)
    wgrid = np.array(wgrid_use)

    Sg = np.interp(wgrid, filt_X, filt_S)

    den = np.trapz(Sg, x=wgrid)
    dens_dict[filt] = den


with open(mag_table_file_2MASS_WISE) as f:
    mag_table_list_2MASS_WISE = f.readlines()

for line in mag_table_list_2MASS_WISE:
    datavec = line.split()
    Teff = float(datavec[0])
    logg = float(datavec[1])
    Jmag = float(datavec[2])
    Hmag = float(datavec[3])
    Kmag = float(datavec[4])
    W1mag = float(datavec[5])
    W2mag = float(datavec[6])
    W3mag = float(datavec[7])
    W4mag = float(datavec[8])

    Teff_logg = (Teff, logg)

    mag_dict_2MASS_WISE[Teff_logg] = [Jmag, Hmag, Kmag, W1mag, W2mag, W3mag, W4mag]

with open(mag_table_file_DENIS) as f:
    mag_table_list_DENIS = f.readlines()

for line in mag_table_list_DENIS:
    datavec = line.split()
    Teff = float(datavec[0])
    logg = float(datavec[1])
    DENIS_imag = float(datavec[2])
    DENIS_Jmag = float(datavec[3])
    DENIS_Kmag = float(datavec[4])

    Teff_logg = (Teff, logg)

    mag_dict_DENIS[Teff_logg] = [DENIS_imag, DENIS_Jmag, DENIS_Kmag]

with open(mag_table_file_Tycho2) as f:
    mag_table_list_Tycho2 = f.readlines()

for line in mag_table_list_Tycho2:
    datavec = line.split()
    Teff = float(datavec[0])
    logg = float(datavec[1])
    BTmag = float(datavec[2])
    VTmag = float(datavec[3])

    Teff_logg = (Teff, logg)

    mag_dict_Tycho2[Teff_logg] = [BTmag, VTmag]
i
with open(mag_table_file_APASS) as f:
    mag_table_list_APASS = f.readlines()

for line in mag_table_list_APASS:
    datavec = line.split()
    Teff = float(datavec[0])
    logg = float(datavec[1])
    Bmag = float(datavec[2])
    Vmag = float(datavec[3])
    gmag = float(datavec[4])
    rmag = float(datavec[5])
    imag = float(datavec[6])

    Teff_logg = (Teff, logg)

    mag_dict_APASS[Teff_logg] = [Vmag, Bmag, gmag, rmag, imag]


def fcn2min(params, x_wav_data999, y_mag_data999, sig_mag_data999):

    wav_2MASS_WISE=[1.235, 1.662, 2.159, 3.3526, 4.6028, 11.5608, 22.0883]
    wav_DENIS = [0.78621, 1.22106, 2.146501]
    wav_Tycho2 = [0.4220, 0.5350]
    wav_APASS = [0.5448, 0.4361, 0.4770, 0.6231, 0.7625]

    #print x_wav_data999
    #print y_mag_data999

    #print params.valuesdict()

    T999 = params['tempxx'].value
    logg = params['logg'].value
    frac_r_d999 = params['frac'].value
    #const999 = (6.95e8 / 3.08e16)

    read_DENIS = params['use_DENIS'].value != 0
    read_Tycho2 = params['use_Tycho2'].value != 0
    read_APASS = params['use_APASS'].value != 0

    wav_use = list(wav_2MASS_WISE)
    
    teff_logg = (T999, logg)

    wise_mags_rstar = mag_dict_2MASS_WISE[teff_logg]
    mags_rstar = list(wise_mags_rstar)

    mag_map_dict = {}

    for i in range(len(wav_2MASS_WISE)):
        mag_map_dict[wav_2MASS_WISE[i]] = wise_mags_rstar[i]

    if read_DENIS:
	#denis_mag_map_dict = {}
	wav_use = wav_use + wav_DENIS

	denis_mags_rstar = mag_dict_DENIS[teff_logg]
	for ent in denis_mags_rstar:
	    mags_rstar.append(ent)

	for i in range(len(wav_DENIS)):
	    mag_map_dict[wav_DENIS[i]] = denis_mags_rstar[i]

    if read_Tycho2:
	tycho2_mag_map_dict = {}
	wav_use = wav_use + (wav_Tycho2)

	tycho2_mags_rstar = mag_dict_Tycho2[teff_logg]
	mags_rstar.append(tycho2_mags_rstar)

	for i in range(len(wav_Tycho2)):
	    mag_map_dict[wav_Tycho2[i]] = tycho2_mags_rstar[i]

    if read_APASS:
        wav_use = wav_use + (wav_APASS)
        
        apass_mags_rstar = mag_dict_APASS[teff_logg]
        mags_rstar.append(apass_mags_rstar)

        for i in range(len(wav_APASS)):
            mag_map_dict[wav_APASS[i]] = apass_mags_rstar[i]

    wav = sorted(wav_use)

    mags_ordered = []

    #print mag_map_dict

    #print wav

    for wavelength in wav:
        #print wavelength
	#print mag_map_dict[wavelength]
	mags_ordered.append(mag_map_dict[wavelength])

    mags_rstar_use = np.zeros(y_mag_data999.size)
    mags_d = np.zeros(y_mag_data999.size)

    for i in range(y_mag_data999.size):
        mags_rstar_use[i] = mags_ordered[i]

    for i in range(y_mag_data999.size):
        mags_d[i] = mags_rstar_use[i] - (5.*np.log10(frac_r_d999))

    #print T999, logg, mags_rstar_use

    #spect_file = 'BT-Settl_M-0.0a+0.0/lte'+teff_pull+'.0-'+logg_pull+'-0.0a+0.0.BT-Settl.spec.7'

    #x_wav999 = x_wv_data999*(1.e-6)
    #x_wav_angstroms = x_wv_data999*(1.e4)
    #c_angstroms = c*(1.e10)

    #frequency999 = c/x_wav999
    #num_filts = len(x_wav_data999.tolist())

    #b_nu999 = (2*h*(frequency999**3)) / (((c)**2) * (np.exp(h*frequency999 / (k*T999)) - 1))
    #factor999 = frac_r_d999**2

    #f_nu_999_wrong_units = fluxdrive_star(spect_file, factor999, num_filts)
    #f_nu_999 = np.zeros(num_filts)

    #for i in range(num_filts):
    #    f_nu_999[i] = f_nu_999_wrong_units[i] * x_wav_angstroms[i] * x_wav_angstroms[i] / c_angstroms

    #f_nu999 = pi*b_nu999*factor999
    #j_f_nu999 = f_nu999*frequency999	
    #model999 = f_nu_999 
    
    #residual = model999 - y_flux_data999

    model999 = mags_d

    #print y_mag_data999

    residual = model999 - y_mag_data999
    weighted = np.sqrt(np.square(residual) / np.square(sig_mag_data999))


    #weighted = np.zeros(y_mag_data999.size)

    #print y_flux

    #for i in range(y_flux_data999.size):
    #    weighted[i] = np.sqrt((residual[i]**2)/(sig_mag_data999[i]**2))

    #free_param_count = 0.
    #free_param_name = ""


    #for param in params:
    #    if param.vary:
    #        free_param_count += 1
    #        free_param_name = param.name

    #print weighted

    return weighted

def planck_lambda(wavelength, temperature):

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

def fcn2min1(params, x_disk_data9991, y_disk_data9991, sig_disk_data9991, synth_flux):

    Tdisk = params['temp_disk'].value
    x19991 = params['number_dust_rad_distance'].value

    num_filts = len(x_disk_data9991.tolist())

    spec_X = np.linspace(0.5, 50., 49501)
    spec_X_met = spec_X*1.e-6

    x_disk_data_met = x_disk_data9991*1.e-6

    x_disk_data_angstroms = x_disk_data9991 * 1.e4

    c_angstroms = c*1.e10

    synth_flux_use = synth_flux[-num_filts:]

    tempBB_disk = planck_lambda(spec_X_met, Tdisk)
    factor_disk = x19991

    f_lambda_disk_wrong_units = np.pi * tempBB_disk
    f_lambda_disk_slightly_less_wrong_units = f_lambda_disk_wrong_units * 1.e-7


    #print f_lambda_disk

    #combined_spect = f_lambda_disk + (spec_S*r_d_star*r_d_star)

    #f_nu_9991_wrong_units = fluxdrive_disk(spec_X, f_lambda_disk_slightly_less_wrong_units, factor_disk, num_filts)
    f_nu_9991_wrong_units = fluxdrive_disk_test(filter_dict, dens_dict, spec_X, f_lambda_disk_slightly_less_wrong_units, factor_disk, num_filts)
    #f_nu_9991 = np.zeros(num_filts)

    #print f_nu_9991_wrong_units

    #for i in range(num_filts):
    #    f_nu_9991[i] = f_nu_9991_wrong_units[i] * spec_X[i] * spec_X[i] / c_angstroms

    #c_angstroms = c*1.e10

    f_nu_9991_slightly_less_wrong_units = np.zeros(num_filts)

    #print f_nu_9991_slightly_less_wrong_units

    for i in range(num_filts):
        #print i
        f_nu_9991_slightly_less_wrong_units[i] = f_nu_9991_wrong_units[i] * ((x_disk_data_angstroms[i])**2) / c_angstroms 

    #f_nu_9991_slightly_less_wrong_units = (f_nu_9991_wrong_units * np.square(x_disk_data_angstroms)) / c_angstroms

    #print f_nu_9991_slightly_less_wrong_units

    f_nu_9991 = f_nu_9991_slightly_less_wrong_units * 1.e-3

    #combined = bnu_disk_data_slightly_less_wrong_units + synth_flux_use

    #model_9991 = fnu_disk_data + synth_flux_use

    model_9991 = f_nu_9991 + synth_flux_use

    #print f_nu_9991
    #print synth_flux_use

    residual = model_9991 - y_disk_data9991

    #weighted = np.sqrt(np.square(residual) / np.square(sig_disk_data9991))

    #weighted = np.zeros(y_disk_data9991.size)

    #print residual

    #for i in range(y_disk_data9991.size):
    #    weighted[i] = np.sqrt((residual[i]**2)/(sig_disk_data9991[i]**2))

    #print weighted

    #free_param_count = 0.
    #free_param_name = ""

    #for param in params:
    #    if param.vary:
    #        free_param_count += 1
    #        free_param_name = param.name

    return residual


def count_excess_bands(real_fluxes, synth_fluxes, errors):

    difs = np.array(real_fluxes) - np.array(synth-fluxes)

    sigmas = difs/np.array(errors)

    num_sig_excesses = 0
    for i in range(sigmas.size):
	if sigmas > 5:
	    num_sig_excesses += 1

    return num_sig_excesses
    


def find_errors_final(valvec, chivec):
    chi_coeffs = np.polyfit(valvec, chivec, 2)

    valmin = (-chi_coeffs[1])/(2.*chi_coeffs[0])
    chi_min = (chi_coeffs[0]*valmin*valmin) + (chi_coeffs[1]*valmin) + chi_coeffs[2]

    chi_min_plusone = chi_min + 1.

    A = chi_coeffs[0]
    B = chi_coeffs[1]
    C = chi_coeffs[2] - chi_min_plusone

    error_one = (-B + np.sqrt((B**2) - (4.*A*C)))/(2.*A)
    error_two = (-B - np.sqrt((B**2) - (4.*A*C)))/(2.*A)

    if error_one < error_two:
        pos_error = error_two
        neg_error = error_one
    else:
        pos_error = error_one
        neg_error = error_two

    pos_error_val = pos_error - valmin
    neg_error_val = neg_error - valmin

    return (pos_error_val, neg_error_val)


####f_disk1 = open('testing_rnd2.txt','w')
wise_val = []
with open ('input_ALMA2017b.txt','r') as f:                    #WISE_2MASS input FILE##########################################
    for line in f:
        line = str(line)
        el_line = line.split()
        line_len = len(el_line)
        if (line_len < 15 and line_len > 0):
##                f_disk1.write('%s\n' %(el_line))
                print ('back1')
        else:
                wise_val.append(el_line)
        count = count + 1
#print (count)
wise_val_mod2 = [x for x in wise_val if x != []]
#print (wise_val_mod2)
###%%%%%%%%%%%%%%%% indexing WISE ids for WISE+2MASS data
index_wise_wiseid = []
for el_wiseid in range(len(wise_val_mod2)):
        index_wise_wiseid.append(wise_val_mod2[el_wiseid][0])	
#print (index_wise_wiseid)
#################################################################flux calculation normal/WISE+2MASS
wav_2MASS_WISE=[1.235, 1.662, 2.159, 3.3526, 4.6028, 11.5608, 22.0883]
band_2MASS_WISE=[0.162, 0.251, 0.262, 0.66256, 1.0423, 5.5069, 4.1013]
zer_2MASS_WISE=list(np.array([1594, 1024, 666.7, 309.540, 171.787, 31.674, 8.363])*1.e-26)
zer_lambda_2MASS_WISE = [3.129e-13, 1.133e-13, 4.283e-14, ]
#f_sed_info = open('sed_info2.txt','w')
flux_wise_mass2 = []
nsig=1.0 
finally_itr = 0
other_information = []
w1_w3_uncert = []
w1_w4_uncert = []

curtime = time.time()

f1 = open('Fitting_outputs_run_'+str(curtime)+'.dat','w')

wav_DENIS = [0.78621, 1.22106, 2.146501]
zer_lambda_DENIS = [1.18194e-9, 3.2005e-10, 4.34266e-11]
zer_DENIS = [2.437e-23, 1.592e-23, 6.674e-24]

wav_Tycho2 = [0.4220, 0.5350]
zer_lambda_Tycho2 = [6.798e-9, 4.029e-9]
zer_Tycho2 = [4.038e-23, 3.847e-23]

wav_APASS = [0.5448, 0.4361, 0.4770, 0.6231, 0.7625]
zer_lambda_APASS = [3.677e-9, 5.738e-9, 4.784e-9, 2.804e-9, 1.872e-9]
zer_APASS = [3.640e-23, 4.260e-23, 3.631e-23, 3.631e-23, 3.631e-23]


zeros_dict = {}

for i in range(len(wav_2MASS_WISE)):
    zeros_dict[wav_2MASS_WISE[i]] = zer_2MASS_WISE[i]

for i in range(len(wav_DENIS)):
    zeros_dict[wav_DENIS[i]] = zer_DENIS[i]

for i in range(len(wav_Tycho2)):
    zeros_dict[wav_Tycho2[i]] = zer_Tycho2[i]	

for i in range(len(wav_APASS)):
    zeros_dict[wav_APASS[i]] = zer_APASS[i]

for el_flux_calc in range(len(wise_val_mod2)):
        designation = wise_val_mod2[el_flux_calc][0]
        w1mag = float(wise_val_mod2[el_flux_calc][1])
        w1mag = w1mag - (-0.1359+0.0396*w1mag-0.0023*w1mag*w1mag)      #correction
        w1err = float(wise_val_mod2[el_flux_calc][2])
        w2mag = float(wise_val_mod2[el_flux_calc][3])
        if (w2mag <= 8):
        	
                w2mag = w2mag - (-0.3530 + 0.8826 *w2mag - 0.2380 *w2mag*w2mag + 0.0170*w2mag*w2mag*w2mag)
        else :
                w2mag = w2mag
        w2err = float(wise_val_mod2[el_flux_calc][4])
        w3mag = float(wise_val_mod2[el_flux_calc][5])
        w3err = float(wise_val_mod2[el_flux_calc][6])
        w4mag = float(wise_val_mod2[el_flux_calc][7])
        w4err = float(wise_val_mod2[el_flux_calc][8])
        w1_w3_diff = w1mag - w3mag
        w1_w3_uncrtnty = (w1err**2 + w3err**2)**0.5
##        w1_w3_uncert.append(w1_w3_uncrtnty)
        w1_w4_diff = w1mag - w4mag
        w1_w4_uncrtnty = (w1err**2 + w4err**2)**0.5
##        w1_w4_uncert.append(w1_w4_uncrtnty)
        jmag = float(wise_val_mod2[el_flux_calc][9])
        jmagerr =  float(wise_val_mod2[el_flux_calc][10])
        hmag = float(wise_val_mod2[el_flux_calc][11])
        hmagerr =  float(wise_val_mod2[el_flux_calc][12])
        kmag = float(wise_val_mod2[el_flux_calc][13])
        kmagerr =  float(wise_val_mod2[el_flux_calc][14])

	wise_mags = [jmag, hmag, kmag, w1mag, w2mag, w3mag, w4mag]
	wise_magerrs = [jmagerr, hmagerr, kmagerr, w1err, w2err, w3err, w4err]

	wave_mag_dict = {}
	for i in range(len(wise_mags)):
	    wave_mag_dict[wav_2MASS_WISE[i]] = (wise_mags[i], wise_magerrs[i])

	use_DENIS = False
	use_Tycho2 = False
        use_APASS = False

	denis_dict = {}
	tycho2_dict = {}
        apass_dict = {}

        if len(wise_val_mod2[el_flux_calc]) > 15:
	    add_source = wise_val_mod2[el_flux_calc][15]
	    if (add_source == 'DENIS'):
		use_DENIS = True
	        denis_imag = float(wise_val_mod2[el_flux_calc][16])
		denis_imagerr = float(wise_val_mod2[el_flux_calc][17])
		denis_jmag = float(wise_val_mod2[el_flux_calc][18])
		denis_jmagerr = float(wise_val_mod2[el_flux_calc][19])
		denis_kmag = float(wise_val_mod2[el_flux_calc][20])
		denis_kmagerr = float(wise_val_mod2[el_flux_calc][21])
		denis_mags = [denis_imag, denis_jmag, denis_kmag]
		denis_magerrs = [denis_imagerr, denis_jmagerr, denis_kmagerr]
		for i in range(len(wav_DENIS)):
		    wave_mag_dict[wav_DENIS[i]] = (denis_mags[i], denis_magerrs[i])

	    elif (add_source == 'Tycho2'):
		use_Tycho2 = True
	        BTmag = float(wise_val_mod2[el_flux_calc][16])
		BTmagerr = float(wise_val_mod2[el_flux_calc][17])
		VTmag = float(wise_val_mod2[el_flux_calc][18])
		VTmagerr = float(wise_val_mod2[el_flux_calc][19])
		tycho2_mags = [BTmag, VTmag]
		tycho2_magerrs = [BTmagerr, VTmagerr]
		for i in range(len(wav_Tycho2)):
		    wave_mag_dict[wav_Tycho2[i]] = (tycho2_mags[i], tycho2_magerrs[i])

            elif (add_source == 'APASS'):
                use_APASS = True
		Vmag = float(wise_val_mod2[el_flux_calc][16])
		Vmagerr = float(wise_val_mod2[el_flux_calc][17])
		Bmag = float(wise_val_mod2[el_flux_calc][18])
		Bmagerr = float(wise_val_mod2[el_flux_calc][19])
		gmag = float(wise_val_mod2[el_flux_calc][20])
		gmagerr = float(wise_val_mod2[el_flux_calc][21])
		rmag = float(wise_val_mod2[el_flux_calc][22])
		rmagerr = float(wise_val_mod2[el_flux_calc][23])
		imag = float(wise_val_mod2[el_flux_calc][24])
		imagerr = float(wise_val_mod2[el_flux_calc][25])
		apass_mags = [Vmag, Bmag, gmag, rmag, imag]
		apass_magerrs = [Vmagerr, Bmagerr, gmagerr, rmagerr, imagerr]
		for i in range(len(wav_APASS)):
		    wave_mag_dict[wav_APASS[i]] = (apass_mags[i], apass_magerrs[i])
	print use_APASS		

	if (use_Tycho2 and use_DENIS):
	    wav_temp = wav_2MASS_WISE + wav_Tycho2 + wav_DENIS
	    #mags = wise_mags + tycho2_mags + denis_mags
	    #magerrs = wise_magerrs + tycho2_magerrs + denis_magerrs
	elif (use_Tycho2):
	    wav_temp = wav_2MASS_WISE + wav_Tycho2
	    #mags = wise_mags + tycho2_mags
	    #magerrs = wise_magerrs + tycho2_magerrs
	elif (use_DENIS):
	    wav_temp = wav_2MASS_WISE + wav_DENIS
	    #mags = wise_mags + denis_mags
	    #magerrs = wise_magerrs + denis_magerrs
	elif (use_APASS):
	    wav_temp = wav_2MASS_WISE + wav_APASS
	else:
	    wav_temp = wav_2MASS_WISE
	    #mags = wise_mags
	    #magerrs = wise_magerrs

	wav = sorted(wav_temp)
	mags = []
	magerrs = []
	zer = []
	for wavelength in wav:
	    cur = wave_mag_dict[wavelength]
	    mags.append(cur[0])
	    magerrs.append(cur[1])

	    cur1 = zeros_dict[wavelength]
	    zer.append(cur1)

        np_wav = np.array(wav)
        #np_band = np.array(band)
        np_zer = np.array(zer)
        np_mags = np.array(mags)
        np_magerrs = np.array(magerrs)
        fluxes=np_zer*(10.0**(-0.4*np_mags)) 
        fluxerrhigh=10**(-0.4*np_mags+nsig*np_magerrs)*np_zer
        fluxerrlow=10**(-0.4*np_mags-nsig*np_magerrs)*np_zer
        fluxerrshigh=fluxerrhigh-fluxes
        fluxerrslow=fluxes-fluxerrlow
        nufnu_flux = c/(np_wav*10**-6)
	#print np_wav*10**-6
        nu_flux = nufnu_flux*fluxes#*10**-26
        #print (nu_flux)
	#print nufnu_flux * (fluxes*10**-26)
        nu_flux_error_high = nufnu_flux* fluxerrshigh
        nu_flux_error_low = nufnu_flux* fluxerrslow
        flux_list = list(nu_flux)
        nu_flux_error_high_list = list(nu_flux_error_high)
        nu_flux_error_low_list = list(nu_flux_error_low)
        nu_flux_err_tot = np.array([nu_flux_error_high_list,nu_flux_error_low_list])

#        print (flux_list_id)

        print designation

        finally_itr = finally_itr + 1
#        if ((w1mag-w4mag) < 0.25):
#               differ_W1_W4 =  w1mag-w4mag
#               f_w1_w4.write('\n Designation ->%s\t,W1-W4 = %f\n' %(designation,differ_W1_W4)) 
#f_sed_info.write('%s' %str(flux_wise_mass2))

#print ('end is near1')
##################### Power Law time
## code ref: - http://stackoverflow.com/questions/10181151/trying-to-get-reasonable-values-from-scipy-powerlaw-fit
## http://scipy-cookbook.readthedocs.org/items/FittingData.html

        fluxes_all_pl = flux_list 
        wv_mod_pl = wav_2MASS_WISE
        power_law_flux = fluxes_all_pl[-4:]
        power_law_wv = wv_mod_pl[-4:]
        np_power_law_flux = np.array(power_law_flux) 
        np_power_law_wv = np.array( power_law_wv)*1e-06
        xdata_pl = np_power_law_wv
        ydata_pl = np_power_law_flux
        logx = log10(xdata_pl)
        logy = log10(ydata_pl)
        fitfunc_pl = lambda pc, xc: pc[0] + pc[1] * xc
        errfunc_pl = lambda pc, xc, yc: (yc - fitfunc_pl(pc, xc))
        pinit_pl = [1.0, -1.0]
        outc = optimize.leastsq(errfunc_pl,pinit_pl, args=(logx, logy), full_output=1)
        pfinalc = outc[0]
        alphac = pfinalc[1]
        kc = 10.0**pfinalc[0]
#        print (alphac)
        if (alphac >=  0.3):
                class_alpha = 'Class I'
#                f_yso1.write('\n designation = %s\t,alpha = %0.2f\n' %(designation,alphac))
        elif (0.3 > alphac >= -0.3):
                class_alpha = 'Flat Spectrum'
#                f_ysoflat.write('\n designation = %s\t,alpha = %0.2f\n' %(designation,alphac))
        elif (-0.3 > alphac >= - 1.6):
                class_alpha = 'Class II'
#                f_yso2.write('\n designation = %s\t,alpha = %0.2f\n' %(designation,alphac))
        elif (alphac < -1.6):
                class_alpha = 'Class III'
#                f_yso3.write('\n designation = %s\t,alpha = %0.2f\n' %(designation,alphac))
        else :
                class_alpha = 'Undetectable'


############################################## Stellar model atmosphere fitting
	print "Starting atmosphere fit"

        if (use_Tycho2):
	    tycho2_mag_mapping = {}

	if (use_DENIS):
	    denis_mag_mapping = {}

        if (use_APASS):
	    apass_mag_mapping = {}

        wise_mag_mapping = {}

        normalized_val = np.array(nu_flux)/nu_flux[0]
#        print ('pl',normalized_val)
        dumb_func = [0]
        list_new_cut_norma =list(normalized_val) 
        new_cut_norma = list_new_cut_norma[1:]
        check_el_start = dumb_func + new_cut_norma
        check_el_end = new_cut_norma + dumb_func
        val_diff_norma = np.array(check_el_start)- np.array(check_el_end)
        list_val_diff_norma = list(val_diff_norma)
        mod_list_val_diff_norma = list_val_diff_norma[1:]
        count_negative = sum(1 for number in mod_list_val_diff_norma if number < 0)
        if (list_new_cut_norma[0] <= list_new_cut_norma[1]and count_negative == 0):
                wv_mod_new_val = np.array(wav[1:-3])
		mag_mod_new_val = np_mags[1:-3]
		mag_mod_new_val_err = np_magerrs[1:-3]
                flx_mod_new_val = (fluxes[1:-3])
		flx_mod_new_val_err = (fluxerrshigh[1:-3])

        if (list_new_cut_norma[0] <= list_new_cut_norma[1] and count_negative > 0):
	        wv_mod_new_val = np.array(wav[:-3])
		mag_mod_new_val = np_mags[:-3]
		mag_mod_new_val_errs = np_magerrs[:-3]
	        flx_mod_new_val = (fluxes[:-3])        
 		flx_mod_new_val_err = (fluxerrshigh[:-3])

        if (list_new_cut_norma[0] > list_new_cut_norma[1]  and count_negative == 0):
	        wv_mod_new_val = np.array(wav[:-3])
		mag_mod_new_val = np_mags[:-3]
		mag_mod_new_val_err = np_magerrs[:-3]
	        flx_mod_new_val = (fluxes[:-3])
                flx_mod_new_val_err = (fluxerrshigh[:-3])

        if (list_new_cut_norma[0] > list_new_cut_norma[1] and count_negative > 0):
	        wv_mod_new_val = np.array(wav[:-3])
		mag_mod_new_val = np_mags[:-3]
		mag_mod_new_val_err = np_magerrs[:-3] 
	        flx_mod_new_val = (fluxes[:-3])
                flx_mod_new_val_err = (fluxerrshigh[:-3])

        if (list_new_cut_norma[0] > list_new_cut_norma[1]  and count_negative == 0):
	        wv_mod_new_val = np.array(wav[1:-3])
		mag_mod_new_val = np_mags[1:-3]
		mag_mod_new_val_err = np_magerrs[1:-3]
	        flx_mod_new_val = (fluxes[1:-3])
		flx_mod_new_val_err = (fluxerrshigh[1:-3])
                
#        print (wv_mod_new_val)
#        print (flx_mod_new_val)
#        time.sleep(20)
        x_wv_data999 =  np.array(wav[:-4])
	y_mag_data999 = np_mags[:-4]
	sig_mag_data999 = np_magerrs[:-4]
        y_flux_data999 = flx_mod_new_val
        sig_flux_data999 = flx_mod_new_val_err
        x_wav999 = x_wv_data999 *(10**-6)
#        print (x_wav999)

        frequency999 = (c/x_wav999)
#        print (frequency999)

        teff_vec = np.array(range(25, 71))*100.
        logg_vec = np.array([2.5, 3., 3.5, 4., 4.5, 5., 5.5])

        #dist_grid = np.zeros((59, 7))
        r_d_star_grid = np.zeros((46, 7))
        chi_array = np.zeros((46, 7))

	#print teff_vec.size

	#print x_wv_data999
	#print y_mag_data999

        for i in range(46):
            for j in range(7):
                teff = teff_vec[i]
		logg = logg_vec[j]


       		params = Parameters()
        	params.add('tempxx',  value = teff, vary=False)
		params.add('logg', value = logg, vary=False)
        	params.add('frac', value = 2.e-10)
		params.add('use_DENIS', value = 0., vary = False)
		if (use_DENIS):
		    params['use_DENIS'].value = 1.
		params.add('use_Tycho2', value = 0., vary = False)
		if (use_Tycho2):
		    params['use_Tycho2'].value = 1.
		params.add('use_APASS', value = 0., vary = False)
		if (use_APASS):
		    params['use_APASS'].value = 1.
		
#params.add('distance',value = 100)

# do fit, here with leastsq model
        	result99 = minimize(fcn2min, params, args=(x_wv_data999, y_mag_data999, sig_mag_data999))
# calculate final result

        	#residual99 = np.zeros(result99.residual.size)

		#print result99.residual.size

		#print residual99

        	#for k in range(result99.residual.size):
            	#    residual99[k] = np.sqrt((result99.residual[k]**2) * (sig_mag_data999[k]**2))

		weighted99 = result99.residual
		residual99 = np.sqrt(np.square(weighted99) * np.square(sig_mag_data999))

        	final_data1999 = y_mag_data999 + residual99

        	#temp_star_999 = result99.params.get('tempxx').value
        	r_d_star99 = result99.params.get('frac').value
		#print 'teff', teff
		#print 'logg', logg
		#print 'r_d_star99', r_d_star99
		#print 'chisqr', result99.chisqr
        	#logg_star999 = result99.params.get('logg').value

		r_d_star_grid[i][j] = r_d_star99
		#print r_d_star_grid[i][j]
		chi_array[i][j] = result99.chisqr

        print chi_array
	min_pos_chi_array = np.argmin(chi_array)
        min_chi = np.amin(chi_array)
	#print min_chi, min_pos_chi_array

        teff_coord = np.argmin(chi_array) / chi_array.shape[1]
        logg_coord = np.argmin(chi_array) % chi_array.shape[1]

        temp_star_999 = teff_vec[teff_coord]
	logg_star_999 = logg_vec[logg_coord]
        r_d_star_999 = r_d_star_grid[teff_coord][logg_coord]

	#teff_pull = 
        
	#spect_file = 'BT-Settl_M-0.0a+0.0/lte'+teff_pull+'.0-'+logg_pull+'-0.0a+0.0.BT-Settl.spec.7'

	#spectname = 

        #print temp_star_999, logg_star_999, r_d_star_999


#calculate Teff errors
        #dpar = [1., 0., 0.]
	#par = [temp_star_999, logg_star_999, r_d_star_999]
        #Teff_chis = find_errors_spect(fcn2min, x_wv_data999, y_flux_data999, sig_flux_data999, par, dpar, teff_vec)
	Teff_chis = np.zeros(teff_vec.size)

	for i in range(46):
       	    params = Parameters()
            params.add('tempxx',  value = teff_vec[i], vary=False)
	    params.add('logg', value = logg_star_999, vary=False)
            params.add('frac', value = r_d_star_999, vary=False)
	    params.add('use_DENIS', value = 0., vary = False)
	    if (use_DENIS):
		params['use_DENIS'].value = 1.
            params.add('use_Tycho2', value = 0., vary = False)
	    if (use_Tycho2):
		params['use_Tycho2'].value = 1.
            params.add('use_APASS', value = 0., vary = False)
	    if (use_APASS):
		params['use_APASS'].value = 1.

            result991 = minimize(fcn2min, params, args=(x_wv_data999, y_mag_data999, sig_mag_data999))

 	    residual991 = np.sqrt(np.square(result991.residual) * np.square(sig_mag_data999))

	    chisqr991_num = np.square(residual991)
	    chisqr991_den = y_mag_data999 + residual991

	    chisqr991 = np.sum(chisqr991_num/chisqr991_den)

	    Teff_chis[i] = chisqr991

        Teff_errs = find_errors_final(teff_vec, Teff_chis)

#calculate logg errors
        #dpar = [0, 1., 0.]
        #logg_chis = find_errors_spect(fcn2min, x_wv_data999, y_flux_data999, sig_flux_data999, par, dpar, logg_vec)
	logg_chis = np.zeros(logg_vec.size)

        for i in range(7):
	    params = Parameters()
	    params.add('tempxx', value = temp_star_999, vary = False)
	    params.add('logg', value = logg_vec[i], vary = False)
	    params.add('frac', value = r_d_star_999, vary = False)
	    params.add('use_DENIS', value = 0., vary = False)
	    if (use_DENIS):
	        params['use_DENIS'].value = 1.
	    params.add('use_Tycho2', value = 0., vary = False)
	    if (use_Tycho2):
	        params['use_Tycho2'].value = 1.
	    params.add('use_APASS', value = 0., vary = False)
	    if (use_APASS):
	        params.add('use_APASS', value = 1., vary = False)

	    result992 = minimize(fcn2min, params, args=(x_wv_data999, y_mag_data999, sig_mag_data999))
 	    
	    residual992 = np.sqrt(np.square(result992.residual) * np.square(sig_mag_data999))

	    chisqr992_num = np.square(residual992)
	    chisqr992_den = y_mag_data999 + residual992

	    chisqr992 = np.sum(chisqr992_num/chisqr992_den)

	    logg_chis[i] = chisqr992

	logg_errs = find_errors_final(logg_vec, logg_chis)

#calculate r_d errors
        r_d_vec = np.zeros(201)

        r_d_vec_min = 0.5*r_d_star_999
        r_d_vec_max = 2.*r_d_star_999

        #print r_d_vec_min, r_d_vec_max

        for i in range(101):
            r_d_vec[i] = r_d_vec_min + (0.01*i*r_d_vec_min)

        for i in range(100):
            r_d_vec[i+101] = r_d_star_999 + 0.01*i*r_d_star_999

        #dpar = [0., 0., 1.]
        #rd_chis = find_errors_spect(fcn2min, x_wv_data999, y_flux_data999, sig_flux_data999, par, dpar, r_d_vec)
	r_d_chis = np.zeros(r_d_vec.size)
	
	for i in range(r_d_vec.size):
	    params = Parameters()
	    params.add('tempxx', value = temp_star_999, vary = False)
	    params.add('logg', value = logg_star_999, vary = False)
	    params.add('frac', value = r_d_vec[i], vary = False)
	    params.add('use_DENIS', value = 0., vary = False)
	    if (use_DENIS):
	        params['use_DENIS'].value = 1.
	    params.add('use_Tycho2', value = 0., vary = False)
	    if (use_Tycho2):
	        params['use_Tycho2'].value = 1.
	    params.add('use_APASS', value = 0., vary = False)
	    if (use_APASS):
	        params['use_APASS'].value = 1.
	    result993 = minimize(fcn2min, params, args=(x_wv_data999, y_mag_data999, sig_mag_data999))

	    residual993 = np.sqrt(np.square(result993.residual) * np.square(sig_mag_data999))

	    chisqr993_num = np.square(residual993)
	    chisqr993_den = y_mag_data999 + residual993

	    chisqr993 = np.sum(chisqr993_num/chisqr993_den)

	    r_d_chis[i] = chisqr993

        #for i in range(r_d_vec.size):
 	#    print i, r_d_vec[i], r_d_chis[i]
	
	rd_errs = find_errors_final(r_d_vec, r_d_chis)


	print "Finished Star Fit"
	print "T_star:", temp_star_999, 'p/m', Teff_errs
	print "log(g):", logg_star_999, 'p/m', logg_errs
	print "r_d_star:", r_d_star_999, 'p/m', rd_errs
############################Disk Blackbody curve fitting
	print "Starting Disk Fit"

        x_disk_data9991 = np.array(wav_2MASS_WISE[-5:])
        y_disk_data9991 = fluxes[-5:]
	sig_disk_data9991 = fluxerrshigh[-5:]
	#print y_disk_data9991
        #print sig_disk_data9991
        #wav_disk = x_disk_data * 1.e-6


	temp_disk_array19991 = np.linspace(40, 400, num=11)
        t_d_val9991 = list(temp_disk_array19991)
	t_n_val9991 = list(np.array([5, 10, 20, 30, 60, 120, 240]) * 1.e-8)
	disk_temp_store9991 = []
	n_value_store9991 = []
	total_itr9991 = 0
	chi_array9991 = []
	itr_el9991 = 0
	just_check9991 = []
	
	#synth_mags = np.array(mag_dict_2MASS_WISE[(temp_star_999, logg_star_999)])

        synth_mags_2MASS_WISE_list = mag_dict_2MASS_WISE[(temp_star_999, logg_star_999)]
	synth_mags_DENIS_list = mag_dict_DENIS[(temp_star_999, logg_star_999)]
	synth_mags_Tycho2_list = mag_dict_DENIS[(temp_star_999, logg_star_999)]
	synth_mags_APASS_list = mag_dict_APASS[(temp_star_999, logg_star_999)]

        synth_mags_list = list(synth_mags_2MASS_WISE_list)
	wav_use_synth_mags = list(wav_2MASS_WISE)

	synth_mags_dict = {}

	for i in range(len(wav_use_synth_mags)):
	    synth_mags_dict[wav_use_synth_mags[i]] = synth_mags_2MASS_WISE_list[i]
	
	if (use_DENIS):
	    synth_mags_list = synth_mags_list + synth_mags_DENIS_list
	    wav_use_synth_mags = wav_use_synth_mags + wav_DENIS
	if (use_Tycho2):
	    synth_mags_list = synth_mags_list + synth_mags_Tycho2_list
	    wav_use_synth_mags = wav_use_synth_mags + wav_Tycho2
	if (use_APASS):
	    synth_mags_list = synth_mags_list + synth_mags_APASS_list
	    wav_use_synth_mags = wav_use_synth_mags + wav_APASS

	for i in range(len(wav_use_synth_mags)):
	    synth_mags_dict[wav_use_synth_mags[i]] = synth_mags_list[i]

	wav_synth_mags = sorted(wav_use_synth_mags)
	synth_mags_use_list = []
	for wavelength in wav_synth_mags:
	    synth_mags_use_list.append(synth_mags_dict[wavelength])

	synth_mags = np.array(synth_mags_use_list)

	synth_mags_dist = synth_mags - (5.*np.log10(r_d_star_999))
        #synth_mags_dist_err = np.sqrt(((5./(r_d_star_999*log(10.)))**2)*(max(r_d_star_errs)**2))

	print synth_mags_dist

        synth_fluxes=np_zer*(10.0**(-0.4*synth_mags_dist))
	print 'synth:', synth_fluxes
	print 'real:', fluxes
	print 'real errs:', fluxerrshigh

        #synth_fluxerrhigh=10**(-0.4*np_mags+nsig*np_magerrs)*np_zer
        #fluxerrlow=10**(-0.4*np_mags-nsig*np_magerrs)*np_zer
        #fluxerrshigh=fluxerrhigh-fluxes
        #fluxerrslow=fluxes-fluxerrlow


        #params = Parameters()

	timer = []

	res_array = np.zeros((77, 5))

	iter_index = 0

	for disk_temp_el9991 in t_d_val9991:
	    for number_el9991 in t_n_val9991:
		print disk_temp_el9991, number_el9991
		tac = time.time()
		params = Parameters()
		params.add('r_d_star', value = r_d_star_999, vary=False)
		params.add('temp_disk', value=disk_temp_el9991, min=50, max=500)
		params.add('number_dust_rad_distance', value = number_el9991, min=0, max=625.*r_d_star_999)
		#params.add('temp_star', value = temp_star_999, vary=False)
		#params.add('logg', value = logg_star_999, vary=False)
		#params.add('r_d_star', value = r_d_star_999, vary=False)


		result9991 = minimize(fcn2min1, params, args=(x_disk_data9991, y_disk_data9991, sig_disk_data9991, synth_fluxes))
		disk_temp_step9991 = result9991.params.get('temp_disk').value
		n_temp_step9991 = result9991.params.get('number_dust_rad_distance').value
		disk_temp_store9991.insert(total_itr9991, disk_temp_step9991)
		n_value_store9991.insert(total_itr9991, n_temp_step9991)

		#weightres = result9991.residual
		#print weightres

		#res = np.zeros(weightres.size)
	
	
		#for i in range(weightres.size):
		#    res[i] = np.sqrt((weightres[i]**2)*(sig_disk_data9991[i]**2))

		#res_array[iter_index,:] = np.sqrt(np.square(result9991.residual) * np.square(sig_disk_data9991))

		final9991 = y_disk_data9991 + result9991.residual

		#print "res:", res
		#print "weightres:", weightres
		#print "y_disk_data:", y_disk_data9991
		#print "final9991:", final9991

		Original9991 = y_disk_data9991
		Expected9991 = final9991

		#print Original9991
		#print Expected9991

		chi_sqr9991 = sig_disk_data9991*(np.square(Expected9991 - Original9991)/Original9991)
		#lst_chi_sqr9991 = list(chi_sqr9991)
		original_chi9991 = np.sum(chi_sqr9991)

		print original_chi9991
		print result9991.chisqr
		chi_array9991.insert(itr_el9991, original_chi9991)
		itr_el9991 += 1
		total_itr9991 += 1
		tac1 = time.time()
		print "Time:", tac1 - tac
		timer.append(tac1 - tac)
		iter_index += 1

	time_avg = sum(timer) / len(timer)
        print "Out of the loops, average time:", time_avg

	min_chi9991 = min(chi_array9991)
	min_chi_index9991 = chi_array9991.index(min_chi9991)

	#print min_chi9991

	optimized_disk_temp9991 = disk_temp_store9991[min_chi_index9991]
	optimized_n_value_store9991 = n_value_store9991[min_chi_index9991]
	result_approach9991 = (min_chi9991 + min_chi9991*0.1)
	new_chi_temp_n_dist9991 = list(zip(chi_array9991, disk_temp_store9991, n_value_store9991))
	new_chi_temp_n_dist9991.sort()
	#print new_chi_temp_n_dist9991
        


        #disk_temp_errors = find_errors_final(disk_temp_store9991, chi_array9991)
	#n_value_errors = find_errors_final(n_value_store9991, chi_array9991)

	new_all9991 = []
	new_all_itr9991 = 0
	print len(new_chi_temp_n_dist9991)
	for el_all9991 in range(len(new_chi_temp_n_dist9991)):
	    if new_chi_temp_n_dist9991[el_all9991][0] <= result_approach9991:
		new_all9991.append(new_chi_temp_n_dist9991[el_all9991])
	print len(new_all9991)
	new_array_el9991 = []
	for all_el_mod9991 in range(len(new_all9991)):
	    temp9991 = new_all9991[all_el_mod9991][1]
	    n_val9991 = new_all9991[all_el_mod9991][2]

	    tnd9991 = (temp9991, n_val9991)
	    new_array_el9991.append(tnd9991)

	new_array_el9991.sort()
	#print new_array_el9991
	final_disk_temp9991 = new_array_el9991[-1][0]
	final_disk_n9991 = new_array_el9991[-1][1]

	#num_excesses = count_excess_bands(params, x_disk_data9991, y_disk_data9991, sig_disk_data9991)
        #num_excesses = count_excess_bands(
	

######Calculate Tdisk errors
	print "Calculating Tdisk errors"
        t_disk_vec = np.zeros(201)

        t_disk_vec_min = 0.5*final_disk_temp9991
        t_disk_vec_max = 2.*final_disk_temp9991

        #print r_d_vec_min, r_d_vec_max

        for i in range(101):
            t_disk_vec[i] = t_disk_vec_min + (0.01*i*final_disk_temp9991)

        for i in range(100):
            t_disk_vec[i+101] = final_disk_temp9991 + (0.02*i*final_disk_temp9991)

        #dpar = [0., 0., 1.]
        #rd_chis = find_errors_spect(fcn2min, x_wv_data999, y_flux_data999, sig_flux_data999, par, dpar, r_d_vec)
	t_disk_chis = np.zeros(t_disk_vec.size)
	
	for i in range(t_disk_vec.size):
	    params = Parameters()
	    params.add('r_d_star', value = r_d_star_999, vary=False)
	    params.add('temp_disk', value=t_disk_vec[i], vary=False)
	    params.add('number_dust_rad_distance', value = final_disk_n9991, vary=False)

	    #params.add('tempxx', value = temp_star_999, vary = False)
	    #params.add('logg', value = logg_star_999, vary = False)
	    #params.add('frac', value = r_d_vec[i], vary = False)
	    #params.add('use_DENIS', value = 0., vary = False)
	    #if (use_DENIS):
	    #    params['use_DENIS'].value = 1.
	    #params.add('use_Tycho2', value = 0., vary = False)
	    #if (use_Tycho2):
	    #    params['use_Tycho2'].value = 1.
	    result99991 = minimize(fcn2min1, params, args=(x_disk_data9991, y_disk_data9991, sig_disk_data9991, synth_fluxes))

	    final99991 = result99991.residual + y_disk_data9991

	    Original99991 = y_disk_data9991
	    Expected99991 = final99991
		
            chi_sqr99991 = sig_disk_data9991*(((Expected99991 - Original99991)**2)/Original99991)
	    original_chi99991 = np.sum(chi_sqr99991)

	    t_disk_chis[i] = original_chi99991

        #for i in range(r_d_vec.size):
 	#    print i, r_d_vec[i], r_d_chis[i]

	print t_disk_chis	
	td_errs = find_errors_final(t_disk_vec, t_disk_chis)

	print "Td errors:", td_errs

######Calculate ndisk errors
	print "Calculating nd errors"
        n_disk_vec = np.zeros(201)

        n_disk_vec_min = 0.5*final_disk_n9991
        n_disk_vec_max = 2.*final_disk_n9991

        #print r_d_vec_min, r_d_vec_max

        for i in range(101):
            n_disk_vec[i] = n_disk_vec_min + (0.01*i*final_disk_n9991)

        for i in range(100):
            n_disk_vec[i+101] = final_disk_n9991 + (0.02*i*final_disk_n9991)

        #dpar = [0., 0., 1.]
        #rd_chis = find_errors_spect(fcn2min, x_wv_data999, y_flux_data999, sig_flux_data999, par, dpar, r_d_vec)
	n_disk_chis = np.zeros(n_disk_vec.size)
	
	for i in range(n_disk_vec.size):
	    params = Parameters()
	    params.add('r_d_star', value = r_d_star_999, vary=False)
	    params.add('temp_disk', value=final_disk_temp9991, vary=False)
	    params.add('number_dust_rad_distance', value = n_disk_vec[i], vary=False)

	    #params.add('tempxx', value = temp_star_999, vary = False)
	    #params.add('logg', value = logg_star_999, vary = False)
	    #params.add('frac', value = r_d_vec[i], vary = False)
	    #params.add('use_DENIS', value = 0., vary = False)
	    #if (use_DENIS):
	    #    params['use_DENIS'].value = 1.
	    #params.add('use_Tycho2', value = 0., vary = False)
	    #if (use_Tycho2):
	    #    params['use_Tycho2'].value = 1.
	    result99992 = minimize(fcn2min1, params, args=(x_disk_data9991, y_disk_data9991, sig_disk_data9991, synth_fluxes))

	    final99992 = result99992.residual + y_disk_data9991

	    Original99992 = y_disk_data9991
	    Expected99992 = final99992
		
            chi_sqr99992 = sig_disk_data9991*(((Expected99992 - Original99992)**2)/Original99992)
	    original_chi99992 = np.sum(chi_sqr99992)

	    n_disk_chis[i] = original_chi99992

        #for i in range(r_d_vec.size):
 	#    print i, r_d_vec[i], r_d_chis[i]
	
	nd_errs = find_errors_final(n_disk_vec, n_disk_chis)



	print "Finished Disk Fit"
######################Calculate Lir/Lstar
	print "Calculating Lir/Lstar"

	T_star = temp_star_999
	T_disk = final_disk_temp9991
	frac_star = r_d_star_999
	n_rad_disk1 = final_disk_n9991
	frac_star_final = frac_star**2

	ratio_disk_star = n_rad_disk1 / frac_star_final
	ratio_temp = (T_disk/T_star)
	ratio_temp_raised = ratio_temp**4

	f90 = ratio_disk_star * ratio_temp_raised

	print ('T_star', T_star, Teff_errs)
	print ('logg', logg, logg_errs)
	print ('r_d_star', r_d_star_999, rd_errs)
	print ('T_disk', T_disk, td_errs)
	print ('n_rad_disk1', final_disk_n9991, nd_errs)

	T_star_err_use = max(Teff_errs)
	r_d_star_err_use = max(rd_errs)
	T_disk_err_use = max(td_errs)
	n_rad_disk1_err_use = max(nd_errs)

        front_fact = (T_disk**3)/((T_star**4)*(r_d_star_999**2))

	sqrtpart1 = 16.*(n_rad_disk1**2)*(T_disk_err_use**2)
	sqrtpart2 = ((16.*(T_disk**2)*(n_rad_disk1**2))/(T_star**2))*(T_star_err_use**2)
	sqrtpart3 = (T_disk**2)*(n_rad_disk1_err_use**2)
	sqrtpart4 = ((4.*(T_disk**2)*(n_rad_disk1**2))/(frac_star**2))*(r_d_star_err_use**2)

	f90_err = front_fact*np.sqrt(sqrtpart1+sqrtpart2+sqrtpart3+sqrtpart4)

	print ('f90', f90, f90_err)

###########################Store determined values


        final_list = []
        final_list = final_list + wise_val_mod2[el_flux_calc][:]

        final_list.append(temp_star_999)
	final_list.append(logg_star_999)
	final_list.append(r_d_star_999)
	final_list.append(T_disk)
	final_list.append(n_rad_disk1)

	final_list_writer = ' '.join(str(ent) for ent in final_list)

        #f1.write(str(final_array[i][0])+' '+str(final_array[i][1])+' '+str(final_array[i][2])+' '+str(final_array[i][3]) + '\n')
	f1.write(final_list_writer + '\n')

#############################################plotting time
	print "Plotting"

	#tff_pull = '0'+str(T999)
	logg_pull = str(logg)

	#spect_file = 'BT-Settl_M-0.0a+0.0/lte'+teff_pull+'.0-'+logg_pull+'-0.0a+0.0.BT-Settl.spec.7'

        #spec_X, spec_S, spec_dict = fluxdrive_plot(spect_file)

        teff_pull = '0'+str(int(temp_star_999/100))
        logg_pull = str(logg_star_999)

        spect_file = 'BT-Settl_M-0.0a+0.0/lte'+teff_pull+'.0-'+logg_pull+'-0.0a+0.0.BT-Settl.spec.7'

	spec_X, spec_S, spec_dict = fluxdrive_plot(spect_file,1)

        spec_X_short_sample_max = max(spec_X)/10000.
        print spec_X_short_sample_max

        spec_X_short_log = np.linspace(0., np.log10(999500), 1000.)
        spec_X_short_log_port = 10.**spec_X_short_log

	spec_X_short_microns = 0.005*spec_X_short_log_port
	spec_X_short_angstroms = spec_X_short_microns*10000.
	spec_S_short = np.interp(spec_X_short_angstroms, spec_X, spec_S)

        #xwav_angstroms = spec_X
	#xwav_microns = xwav_angstroms * 1.e-4
	#xwav_meters = xwav_angstroms * 1.e-10

	xwav_angstroms = spec_X_short_angstroms
	xwav_microns = spec_X_short_microns
	xwav_meters = spec_X_short_microns*1.e-6

	spec_freqs = c/xwav_meters

	#spec_S_fixed_units = np.zeros(spec_S.size)

	#for i in range(spec_S_short.size):
	#    spec_S_fixed_units[i] = (spec_S_short[i] * xwav_angstroms[i] * xwav_angstroms[i] / (c*1.e10)) *1.e-26 #I don't know why this factor needs to be here but it does

        #for i in range(spec_S_short.size):

	j_fnu_spec_S = spec_S_short * xwav_angstroms * 1.e-3 * (r_d_star_999**2)

        #print spec_S_fixed_units[999:21999]
	#print fluxes

	#f_nu_spec_S = np.zeros(spec_S_short.size)
	#j_fnu_spec_S = np.zeros(spec_S_short.size)

	#for i in range(spec_S_short.size):
	#    f_nu_spec_S[i] = spec_S_fixed_units[i] * (r_d_star_999**2)
	#    j_fnu_spec_S[i] = spec_freqs[i] * spec_S_fixed_units[i]

        #print j_fnu_spec_S[999:21999]
	#print nu_flux

        #xwav_microns_smooth = 

        b_lambda_disk_plot = planck_lambda(xwav_meters, T_disk)
	#b_nu_disk_plot = b_lambda_disk_plot*((xwav_meters*xwav_meters)/c)

	f_nu_disk_plot = np.pi * n_rad_disk1 * b_lambda_disk_plot

	

	j_f_nu_disk_plot = f_nu_disk_plot * xwav_meters

	j_f_nu_model = j_fnu_spec_S + j_f_nu_disk_plot

        list_model_plot = list(j_f_nu_model)

	#max_el_lst = max(list_model_plot)
	#mal_lim_plot = max_el_list * 10
	#min_el_lst = min(list_model_plot)
	#min_lim_plot = mim_lim_plot

	#print j_fnu_spec_S.size
	#print xwav_microns.size

	plt.figure()
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['ytick.labelsize'] = 20
        params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
        plt.rcParams.update(params)

	ax = plt.gca()
	#plt.gcf.subplots_adjust(left=0.15)
        plt.gcf().subplots_adjust(left=0.18)
        plt.gcf().subplots_adjust(right=0.92)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(top=0.95) 
        p16, = ax.plot(xwav_microns,j_fnu_spec_S*1000.,'b--',linewidth=3.,label="Star Blackbody Model")
        p36, = ax.plot(xwav_microns,j_f_nu_disk_plot*1000.,'g',dashes=[8, 4, 2, 4, 2, 4],linewidth=3.,label="Disk Blackbody Model")
        p26, = ax.plot(xwav_microns, j_f_nu_model*1000.,'black',linewidth=3.5,label="Total Blackbody Model")
        ax.errorbar(np_wav,nu_flux*1000,yerr=nu_flux_err_tot*1000, fmt='p',color = 'red', markersize=14)
        ax.set_xlabel(r'$\mathrm{Wavelength(\mu m)}$',fontsize=24)
        ax.set_ylabel(r'$\mathrm{\nu F_\nu(erg\,s^{-1}\,cm^{-2})}$',fontsize=24)
        time.sleep(4)
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.text(0.075, 0.925, 'b)', va='top', color = 'black', transform=ax.transAxes, fontsize=20)

	maxval = max(nu_flux)
        minval = min(nu_flux)

        logmax = np.log10(maxval)
        logmin = np.log10(minval)

        logminbound = math.floor(logmin) - 1.5
	logmaxbound = math.ceil(logmax)

        if ((logmaxbound - logmax) < 0.5):
	    logmaxbound += 0.5

        minbound = 10**logminbound
	maxbound = 10**logmaxbound

	max_x_poss = np.interp(minbound, j_f_nu_model, xwav_microns)

	log_max_x_poss = np.log10(max_x_poss)

	ceil_log_max_x_poss = math.ceil(log_max_x_poss*10.)

	ceil_max_x_poss = round(10**(ceil_log_max_x_poss), 0)

	if (ceil_max_x_poss < 250.):
            maximum_x = ceil_max_x_poss
	else:
	    maximum_x = 250.

        ax.set_xlim([0.4,1000.])
	#ax.set_ylim([minbound, maxbound])        
	ax.set_ylim([1.e-16, 1.e-9])

        print designation
        print "T_star =", T_star
        print "T_disk =", T_disk
        print "f =", f90
	print "W1 - W3 =", w1_w3_diff, "p/m", w1_w3_uncrtnty
	print "W1 - W4 =", w1_w4_diff, "p/m", w1_w4_uncrtnty

        plt.legend(handles=[p16,p26,p36], loc='upper right',borderpad=.5, labelspacing=.2)
        
        path = os.path.join(designation)
#        print (path)
#        plt.show()
        frmt = '.png'
        total_path = path + frmt
#        print (total_path)
        plt.savefig(total_path, fmt='.png', dpi=1000)
#        plt.show()
        time.sleep(3)       
print ('The END')
          
print ('end')

