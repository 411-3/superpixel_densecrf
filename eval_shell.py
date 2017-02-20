
import shlex, subprocess, numpy, time

def printinpy(filename, test_pic_num):
	time_start = time.time()
	print "\n*********Output**********\n"
	# read list_name 
	pic_list = []
	fr = open(filename)
	for i in range(test_pic_num):
		linestr = fr.readline()
		linestr = linestr.strip('\n')
		pic_list.append(linestr)
	
	# exename  = '/home/xduser/LiHuan/python/test/hello.exe'
	exename0 = '/home/xduser/LiHuan/superpixel_crf/superpixel_crf'
	jpgname0 = '/home/xduser/LiHuan/superpixel_crf/pic/test_lh/pic_jpg/'
	fcnname0 = '/home/xduser/LiHuan/superpixel_crf/pic/test_lh/pic_fcn/'
	outname0 = '/home/xduser/LiHuan/superpixel_crf/pic/'

	model = 0 # 0->3; 1->2_modified;
	for w1 in [5, 10]:
		for theta in [40,50,60]:
			for beta in [3,4,5]:
				for w3 in [3,5]:
					for delta in [3]:
						for j in range(test_pic_num):
							args = []
							jpgname = jpgname0 + pic_list[j] + '.jpg'
							fcnname = fcnname0 + pic_list[j] + '.jpg.score.npy'
							outname = outname0 + `w1`+'_'+`theta`+'_'+`beta`+'_'+`w3`+'_'+`delta`+'/'+ pic_list[j]+'.png'
							exename = exename0 +' '+jpgname+' '+fcnname+' '+outname+' '+`model`+' '+`w1`+' '+`theta`+' '+`beta`+' '+`w3`+' '+`delta`
							args.append(exename)
							#args.append('%d' %model )
							# call crf.exe
							child = subprocess.Popen(args, shell=True)
							child.wait()
							del args[:]
							exename = ''
						print "------------------------------A kind of Params has been tested!\n"			
	print "***********************All w has been tested!\n"
	time_end = time.time()
	print time_end - time_start
	print "s"			

