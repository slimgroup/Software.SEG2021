using FFTW, DSP
function WindowFKSpectrum(dataFreq, datainfo)
	#Input: 
	#dataFreq: one shot gather in Frequency domain (nf,nrx,nry)
    #datainfo: a Dictionary contains info for one shot gather
	#Output:
	# one shot gather in Frequency domain after denoising
    nf,nrx,nry = size(dataFreq)
	#fourier transform along Rx
	Data2 = ifft(dataFreq,2);
	Data_FKx = circshift(Data2,[0 ceil(Int, datainfo["nrx"]/2)-1]);

	#fourier transform along Ry
	Data3 = ifft(Data_FKx,3);
	Data_FKxy = circshift(Data3,[0 0 ceil(Int, datainfo["nry"]/2)-1]);

	Rgrid = datainfo["Rgrid"];
	dt = datainfo["dt"];

	Window = zeros(nf,datainfo["nrx"]);
	Center = Int((datainfo["nrx"]+1)/2);
	Data_NFKxy = copy(Data_FKxy);

	for i = 1:nf  #design band pass filter according to the FK spectrum
		if i >= 240
			low = Center-320;
			high = Center+320;
			band = 5; 
			Window[i,low+band+1:high-band-1] .= DSP.Windows.tukey(high-band-1-(low+band+1)+1,0.2;padding=0,zerophase=false);
		else
			range1 = round(230/239*i+(90-230/239));
			low = Center-Int(range1);
			high = Center + Int(range1);
			band = 5; 
			Window[i,low+band+1:high-band-1] .= DSP.Windows.tukey(high-band-1-(low+band+1)+1,0.2;padding=0,zerophase=false);
		end
	end
	

	for j = 1:nrx
		Data_NFKxy[:,:,j] = Data_NFKxy[:,:,j] .* Window;
		Data_NFKxy[:,j,:] = Data_NFKxy[:,j,:] .* Window;
	end
	
	Data3 = circshift(Data_NFKxy,[0 0 -ceil(Int, datainfo["nry"]/2)+1]);
	DataNFKx = fft(Data3,3);

	Data3 = circshift(DataNFKx,[0 -ceil(Int, datainfo["nrx"]/2)+1]);
	DataFreq = fft(Data3,2);
	
	return DataFreq
end