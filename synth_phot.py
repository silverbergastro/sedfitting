import numpy as np
import matplotlib.pyplot as plt

def import_filter(filtername):

    filterfilename = filtername+"_response.dat"

    filter_xlist = []
    filter_Slist = []

    filter_dict = {}

    with open(filterfilename) as f:
        filterfilelist = f.readlines()

    testline = filterfilelist[0]

    for line in filterfilelist:
	#if (flag1):
            
        #    line_use = line[:6] + ' ' + line[6:18] + ' ' + line[18:]
	#print line_use
        datavec = line.split()
        x = float(datavec[0])
	if ('DENIS' in filtername):
	    x = float(datavec[0])*1.e4
        s = float(datavec[1])
        filter_xlist.append(x)
	filter_Slist.append(s)
        filter_dict[x] = s

    filter_X = np.array(filter_xlist)
    #print filter_X
    filter_S = np.array(filter_Slist)



    return filter_X, filter_S, filter_dict


def import_spectrum(spectname,binsize):

    spectfilename = spectname

    spect_xlist = []
    spect_Slist = []
    spect_dict = {}

    with open(spectfilename) as f:
        spectfilelist = f.readlines()

    testline = spectfilelist[0]
    test_res = [pos for pos, char in enumerate(testline) if char == '-']
    line_start = testline.index('1')

    flag1 = (test_res[0] < (7 + line_start))
    flag2 = (test_res[1] < (20 + line_start))     


    for line in spectfilelist:
	if (flag1 and flag2):
	    line_use = line[:13] + ' ' + line[13:25] + ' ' + line[25:]
	elif (flag1):
	    line_use = line[:13] + ' ' + line[13:]
	elif (flag2):
	    line_use = line[:25] + ' ' + line[25:]
	else:
	    line_use = line

        #line_use = 
	#print line_use
        
	datavec = line_use.split()
	#if len(datavec[0]) > 6:
        #    xstr = datavec[0][0:5]
	#    sstr = datavec[0][6:17]
	#else:
        xstr = datavec[0]
        sstr = datavec[1]
	sstr1 = sstr.replace('D','e')

	x = float(xstr)
	s = float(sstr1)

	#print x, s

        spect_xlist.append(x)
        spect_Slist.append((10**(s-8.)))
        spect_dict[x] = s

    spect_X_binned = []
    spect_S_binned = []
    
    ents_per_bin = binsize*20.

    #for i in range(int(len(spect_xlist)/ents_per_bin)):
	#print i

     #   spect_X_binned.append(spect_xlist[int(20*i*ents_per_bin)])
#	print i, spect_X_binned[-1]

    #for i in range(int(len(spect_xlist)/ents_per_bin)):
    #    spect_X_binned.append(spect_xlist[ents_per_bin*(2*(i+1))/2])
    #	spect_S_set = spect_Slist[(ents_per_bin*i):((ents_per_bin*(i+1))-1)]
    #	spect_S_binned.append(sum(spect_S_set)/ents_per_bin)

    #spect_X_temp = np.array(spect_xlist)
    #spect_S_temp = np.array(spect_Slist)

    spect_X = np.array(spect_xlist)
    spect_S = np.array(spect_Slist)


    #spect_X = np.array(spect_Xlist)

    #spect_S = np.interp(spect_X, spect_X_temp, spect_S_temp)

    return spect_X, spect_S, spect_dict    

def uniq(inputlist):
    seen = set()
    seen_add = seen.add
    return [x for x in inputlist if not (x in seen or seen_add(x))]

def synth_phot(filtername, spectname, dilfact):

    filter_X, filter_S, filter_dict = import_filter(filtername)

    spec_X, spec_S, spec_dict = import_spectrum(spectname)

    #plt.loglog(spec_X, spec_S)
    #plt.show()

    temp_filtx = filter_X.tolist()
    temp_specx = spec_X.tolist()

    wmin = min(filter_X)
    wmax = max(filter_X)

    temp_specx1 = [i for i in temp_specx if i >= wmin]
    temp_specx2 = [i for i in temp_specx1 if i <= wmax]

    #wgrid = np.zeros(filter_X.size + len(temp_specx2)
    
    wgridlist = temp_filtx + temp_specx2

    wgridlist.sort()

    wgrid_use = uniq(wgridlist)

    wgrid = np.array(wgrid_use)

    Sg = np.interp(wgrid, filter_X, filter_S)
    yg_raw = np.interp(wgrid, spec_X, spec_S)
    #print Sg_raw

    #dist_met = distance * 3.085678e16

    #rad_met = radius * 6.957e8

    #dil_fact = (rad_met/dist_met)**2

    yg = yg_raw * dilfact
    #print yg

    numint = yg*Sg
    #print yg

    num = np.trapz(numint, x=wgrid)
    den = np.trapz(Sg, x=wgrid)

    filt_flux = num/den

    filt_lamflam = filt_flux

    #alt_flux = 

    return filt_lamflam

def synth_phot_disk(filtername, spec_X, spec_S, dilfact):

    filter_X, filter_S, filter_dict = import_filter(filtername)
    
    #plt.loglog(spec_X, spec_S)
    #plt.show()

    temp_filtx = filter_X.tolist()
    temp_specx = spec_X.tolist()

    wmin = min(filter_X)
    wmax = max(filter_X)

    temp_specx1 = [i for i in temp_specx if i >= wmin]
    temp_specx2 = [i for i in temp_specx1 if i <= wmax]

    #temp_specx2 = spec_X[spec_X >= wmin and spec_X <= wmax]

    #wgrid = np.zeros(filter_X.size + len(temp_specx2)
    
    wgridlist = temp_filtx + temp_specx2

    wgridlist.sort()

    wgrid_use = uniq(wgridlist)

    wgrid = np.array(wgrid_use)

    Sg = np.interp(wgrid, filter_X, filter_S)
    yg_raw = np.interp(wgrid, spec_X, spec_S)
    #print yg_raw

    #dist_met = distance * 3.085678e16

    #rad_met = radius * 6.957e8

    #dil_fact = (rad_met/dist_met)**2

    yg = yg_raw * dilfact
    #print yg

    numint = yg*Sg
    #print numint
    #print yg

    num = np.trapz(numint, x=wgrid)
    den = np.trapz(Sg, x=wgrid)

    filt_flux = num/den

    filt_lamflam = filt_flux

    #alt_flux =

    #print num, den 

    #print filt_flux

    return filt_lamflam

def synth_phot_disk_test(filt_dict, den, spec_X, spec_S, dilfact):

    #filter_X, filter_S, filter_dict = import_filter(filtername)
    
    filter_X = np.array(sorted(filt_dict.keys()))
    filter_S = np.zeros(filter_X.size)
    for i in range(filter_X.size):
        filter_S[i] = filt_dict[filter_X[i]]

    #plt.loglog(spec_X, spec_S)
    #plt.show()

    temp_filtx = filter_X.tolist()
    temp_specx = spec_X.tolist()

    wmin = min(filter_X)
    wmax = max(filter_X)

    temp_specx1 = [i for i in temp_specx if i >= wmin]
    temp_specx2 = [i for i in temp_specx1 if i <= wmax]

    #temp_specx2 = spec_X[spec_X >= wmin and spec_X <= wmax]

    #wgrid = np.zeros(filter_X.size + len(temp_specx2)
    
    wgridlist = temp_filtx + temp_specx2

    wgridlist.sort()

    wgrid_use = uniq(wgridlist)

    wgrid = np.array(wgrid_use)

    Sg = np.interp(wgrid, filter_X, filter_S)
    yg_raw = np.interp(wgrid, spec_X, spec_S)
    #print yg_raw

    #dist_met = distance * 3.085678e16

    #rad_met = radius * 6.957e8

    #dil_fact = (rad_met/dist_met)**2

    yg = yg_raw * dilfact
    #print yg

    numint = yg*Sg
    #print numint
    #print yg

    num = np.trapz(numint, x=wgrid)
    #den = np.trapz(Sg, x=wgrid)
    #den = 

    #print num

    filt_flux = num/den

    filt_lamflam = filt_flux

    #alt_flux =

    #print num, den 

    #print filt_flux

    return filt_lamflam


def fluxdrive_star(spectname, dil_fact, numfilt):
    nufnu_J = synth_phot('J', spectname, dil_fact)
    nufnu_H = synth_phot('H', spectname, dil_fact)
    nufnu_K = synth_phot('K', spectname, dil_fact)
    nufnu_W1 = synth_phot('W1', spectname, dil_fact)
    nufnu_W2 = synth_phot('W2', spectname, dil_fact)
    nufnu_W3 = synth_phot('W3', spectname, dil_fact)
    nufnu_W4 = synth_phot('W4', spectname, dil_fact)

    #cent_wav_vec = [1.235, 1.662, 2.159, 3.3526, 22.]
    nufnu_vec_temp = [nufnu_J, nufnu_H, nufnu_K, nufnu_W1, nufnu_W2, nufnu_W3, nufnu_W4]

    #if Star:
    nufnu_vec = nufnu_vec_temp[0:numfilt]

    return nufnu_vec

def fluxdrive_disk_test(filt_dicts, dens, diskX, diskS, dil_fact, numfilt):

    K_filtdict = filt_dicts['K']
    W1_filtdict = filt_dicts['W1']
    W2_filtdict = filt_dicts['W2']
    W3_filtdict = filt_dicts['W3']
    W4_filtdict = filt_dicts['W4']
    
    K_den = dens['K']
    W1_den = dens['W1']
    W2_den = dens['W2']
    W3_den = dens['W3']
    W4_den = dens['W4']


    #nufnu_J = synth_phot_disk('J', diskX, diskS, dil_fact)
    #nufnu_H = synth_phot_disk('H', diskX, diskS, dil_fact)
    #print "Starting K"
    nufnu_K = synth_phot_disk_test(K_filtdict, K_den, diskX, diskS, dil_fact)
    #print "Starting W1"
    nufnu_W1 = synth_phot_disk_test(W1_filtdict, W1_den, diskX, diskS, dil_fact)
    #print "Starting W2"
    nufnu_W2 = synth_phot_disk_test(W2_filtdict, W2_den, diskX, diskS, dil_fact)
    #print "Starting W3"
    nufnu_W3 = synth_phot_disk_test(W3_filtdict, W3_den, diskX, diskS, dil_fact)
    #print "Starting W4"
    nufnu_W4 = synth_phot_disk_test(W4_filtdict, W4_den, diskX, diskS, dil_fact)

    #nufnu_vec_temp = [nufnu_J, nufnu_H, nufnu_K, nufnu_W1, nufnu_W2, nufnu_W3, nufnu_W4]

    #nufnu_vec = nufnu_vec_temp[-numfilt:]

    nufnu_vec_temp = [nufnu_K, nufnu_W1, nufnu_W2, nufnu_W3, nufnu_W4]

    nufnu_vec = nufnu_vec_temp[-numfilt:]

    #print nufnu_vec

    return nufnu_vec

def fluxdrive_disk(diskX, diskS, dil_fact, numfilt):
    #nufnu_J = synth_phot_disk('J', diskX, diskS, dil_fact)
    #nufnu_H = synth_phot_disk('H', diskX, diskS, dil_fact)
    nufnu_K = synth_phot_disk('K', diskX, diskS, dil_fact)
    nufnu_W1 = synth_phot_disk('W1', diskX, diskS, dil_fact)
    nufnu_W2 = synth_phot_disk('W2', diskX, diskS, dil_fact)
    nufnu_W3 = synth_phot_disk('W3', diskX, diskS, dil_fact)
    nufnu_W4 = synth_phot_disk('W4', diskX, diskS, dil_fact)

    #nufnu_vec_temp = [nufnu_J, nufnu_H, nufnu_K, nufnu_W1, nufnu_W2, nufnu_W3, nufnu_W4]

    #nufnu_vec = nufnu_vec_temp[-numfilt:]

    nufnu_vec = [nufnu_K, nufnu_W1, nufnu_W2, nufnu_W3, nufnu_W4]

    return nufnu_vec


def fluxdrive_plot(spectname,binsize):
    spec_X, spec_S, spec_dict = import_spectrum(spectname,binsize)

    return spec_X, spec_S, spec_dict

def fluxdrive_Tycho(spec_X, spec_S):
    #spec_X, spec_S, spec_dict = import_spectrum(spectname, 1.)

    flux_BT = synth_phot_disk('BT', spec_X, spec_S, 1.)
    flux_VT = synth_phot_disk('VT', spec_X, spec_S, 1.)

    return flux_BT, flux_VT

def fluxdrive_DENIS(spec_X, spec_S):
    flux_DENIS_I = synth_phot_disk('DENIS_I', spec_X, spec_S, 1.)
    flux_DENIS_J = synth_phot_disk('DENIS_J', spec_X, spec_S, 1.)
    flux_DENIS_Ks = synth_phot_disk('DENIS_Ks', spec_X, spec_S, 1.)

    return flux_DENIS_I, flux_DENIS_J, flux_DENIS_Ks
