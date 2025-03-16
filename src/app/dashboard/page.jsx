'use client';
import { useEffect, useState, useCallback } from 'react';
import { socket } from '@/socket';
import { Maps } from '@/components/maps';
import toast from 'react-hot-toast';
import { AlertTriangle, BarChart3 } from 'lucide-react';
import axios from 'axios';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { getTopCrimeCity } from '@/services/topCrime';
import { crimeByCity } from '../constant';

function Page() {
  const [isConnected, setIsConnected] = useState(false);
  const [transport, setTransport] = useState('N/A');
  const [serverMessage, setServerMessage] = useState('');
  const [processingStatus, setProcessingStatus] = useState('');
  const [videoResults, setVideoResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [total, setTotal] = useState(0);
  const [policeStationInfo, setPoliceStationInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [address, setAddress] = useState('');
  const [crimeCityData, setCrimeCityData] = useState([]);
  const [showCrimeLayer, setShowCrimeLayer] = useState(true);

  const startProcessing = useCallback(() => {
    if (!isProcessing) {
      setVideoResults([]);
      socket.emit('start_processing');
      setIsProcessing(true);
    }
  }, [isProcessing]);

  const stopProcessing = useCallback(() => {
    socket.emit('stop_processing');
    setIsProcessing(false);
  }, []);

  useEffect(() => {
    const onConnect = () => {
      setIsConnected(true);
      setTransport(socket.io.engine.transport.name);
      socket.io.engine.on('upgrade', (transport) => {
        setTransport(transport.name);
      });
    };

    const onDisconnect = () => {
      setIsConnected(false);
      setTransport('N/A');
    };

    const onServerMessage = (data) => {
      setServerMessage(data.message);
    };

    const onProcessingUpdate = (data) => {
      setProcessingStatus(`Processing ${data.current} of ${data.total}`);
      setTotal(data.total);
    };

    const onVideoResult = async (data) => {
      console.log('Video Data ', data);
      setVideoResults((prev) => [data, ...prev]);

      if (data.address) {
        const coords = await geocodeAddress(data.address);
        if (coords) {
          setAddress(coords);
        }
      }
    };

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('server_message', onServerMessage);
    socket.on('processing_update', onProcessingUpdate);
    socket.on('video_result', onVideoResult);

    startProcessing();

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('server_message', onServerMessage);
      socket.off('processing_update', onProcessingUpdate);
      socket.off('video_result', onVideoResult);
    };
  }, [startProcessing]);

  useEffect(() => {
    const topCrimes = getTopCrimeCity(crimeByCity);

    const formattedCrimeData = topCrimes.map((city) => ({
      city: city.city,
      count: city.count,
    }));

    setCrimeCityData(formattedCrimeData);
  }, []);

  const findNearestPoliceStation = async (address) => {
    if (!address) {
      console.error('No address provided');
      return {
        address: 'No address provided',
        phone: 'N/A',
      };
    }

    try {
      setIsLoading(true);

      const apiKey =
        process.env.NEXT_PUBLIC_GEMINI_API_KEY ||
        'AIzaSyDB9gtQuwLTOsAxOQFsfD4I2Hc_QNx6d_A';
      const genAI = new GoogleGenerativeAI(apiKey);

      const model = genAI.getGenerativeModel({
        model: 'gemini-1.5-pro',
      });

      const prompt = `
      Find the nearest police station to this address: ${address}
      
      Use online search to identify the closest police station and provide the following details:
      1. Full address of the police station
      2. Phone number of the police station
      
      Format your response exactly like this:
      Address: [full address]
      Phone Number: [phone number]
      
      Only include these three fields with no additional text, introduction, or explanations.
      `;

      console.log('Searching for police station near:', address);

      const result = await model.generateContent(prompt);
      const response = result.response.text();

      console.log('Raw Gemini response:', response);

      const addressMatch = response.match(/Address:\s*([^\n]+)/i);
      const phoneMatch = response.match(/Phone Number:\s*([^\n]+)/i);

      const info = {
        address: addressMatch ? addressMatch[1].trim() : 'Not found',
        phone: phoneMatch ? phoneMatch[1].trim() : 'Not found',
      };

      console.log('Extracted police station info:', info);
      return info;
    } catch (error) {
      console.error('Error finding police station:', error);
      return {
        address: 'Error finding information',
        phone: 'Error finding information',
      };
    } finally {
      setIsLoading(false);
    }
  };

  const geocodeAddress = async (address) => {
    if (!address) return null;

    const encodedAddress = encodeURIComponent(address);
    const apiKey =
      process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ||
      'AIzaSyCb4aE_i6ANsLCSaGLt3NuUqkDdkcbkV7o';
    const geocodeUrl = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodedAddress}&key=${apiKey}`;

    try {
      const response = await axios.get(geocodeUrl);
      if (response.data.results && response.data.results.length > 0) {
        const { lat, lng } = response.data.results[0].geometry.location;
        return { lat, lng };
      } else {
        console.error('No geocoding results found');
        return null;
      }
    } catch (error) {
      console.error('Geocoding error:', error);
      return null;
    }
  };

  useEffect(() => {
    console.log(isConnected);
    if (videoResults.length > 0 && videoResults.length <= total) {
      toast.dismiss();
      const latestResult = videoResults[0];

      if (!latestResult.address) {
        console.error('No address in the latest video result');
        return;
      }

      findNearestPoliceStation(latestResult.address).then((policeInfo) => {
        setPoliceStationInfo(policeInfo);

        toast.custom(
          (t) => (
            <div
              className={`${
                t.visible ? 'animate-enter' : 'animate-leave'
              } max-w-md w-full bg-red-600 shadow-xl rounded-lg pointer-events-auto flex flex-col ring-1 ring-red-700 text-white`}
            >
              <div className='flex justify-between items-center p-4 border-b border-red-500'>
                <div className='flex items-center'>
                  <AlertTriangle className='h-5 w-5 text-yellow-300 mr-2' />
                  <span className='font-bold'>Emergency Alert</span>
                </div>
                <button
                  onClick={() => toast.dismiss(t.id)}
                  className='text-white hover:text-gray-200'
                >
                  <span className='sr-only'>Close</span>
                  <svg
                    className='h-5 w-5'
                    viewBox='0 0 20 20'
                    fill='currentColor'
                  >
                    <path
                      fillRule='evenodd'
                      d='M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z'
                      clipRule='evenodd'
                    />
                  </svg>
                </button>
              </div>

              {isLoading ? (
                <div className='p-4'>
                  <div className='flex justify-center'>
                    <div className='animate-spin rounded-full h-6 w-6 border-b-2 border-white'></div>
                  </div>
                  <p className='text-sm text-center mt-2'>
                    Searching for nearest police station...
                  </p>
                </div>
              ) : (
                <div className='p-4'>
                  <div className='text-sm space-y-2'>
                    <p>
                      <span className='font-semibold'>Incident Location:</span>{' '}
                      {latestResult.address}
                    </p>
                    <p>
                      <span className='font-semibold'>
                        Nearest Police Station:
                      </span>{' '}
                      {policeInfo.address}
                    </p>
                    <p>
                      <span className='font-semibold'>Contact:</span>{' '}
                      {policeInfo.phone}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ),
          {
            duration: 8000,
          }
        );
      });
    }

    return () => {
      toast.dismiss();
    };
  }, [videoResults, total]);

  const ConnectionStatus = () => (
    <div className='absolute top-4 right-4 z-10 bg-white/90 rounded-md px-3 py-2 shadow-md'>
      <div className='flex items-center gap-2'>
        <div
          className={`w-3 h-3 rounded-full ${
            isConnected ? 'bg-green-500' : 'bg-red-500'
          }`}
        />
        <span
          className={
            isConnected
              ? 'text-sm font-medium text-green-500'
              : 'text-sm font-medium text-red-500'
          }
        >
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </div>
  );

  const CrimeStatsCard = () => {
    const topCrimesList = crimeCityData.slice(0, 5);

    return (
      <div className='absolute top-4 left-4 z-10 bg-white/90 rounded-md px-3 py-2 shadow-md max-w-xs'>
        <div className='flex items-center gap-2 mb-2'>
          <BarChart3 className='h-4 w-4 text-red-600' />
          <span className='text-sm font-medium text-gray-800'>
            Top Crime Hotspots
          </span>
        </div>
        <div className='max-h-40 overflow-y-auto text-xs text-gray-800'>
          {topCrimesList.map((city, index) => (
            <div
              key={index}
              className='flex justify-between items-center py-1 border-b border-gray-100 last:border-0'
            >
              <span>{city.city}</span>
              <span className='font-medium text-red-600'>{city.count}</span>
            </div>
          ))}
        </div>
        <button
          onClick={() => setShowCrimeLayer(!showCrimeLayer)}
          className='w-full mt-2 text-xs py-1 px-2 bg-blue-50 hover:bg-blue-100 text-blue-600 rounded transition'
        >
          {showCrimeLayer ? 'Hide Crime Layer' : 'Show Crime Layer'}
        </button>
      </div>
    );
  };

  return (
    <div className='h-screen w-screen bg-amber-100 flex flex-col'>
      <ConnectionStatus />
      {/* <CrimeStatsCard /> */}

      <div className='flex-1 p-4'>
        <div className='w-full h-full rounded-lg overflow-hidden border border-gray-300 shadow-lg'>
          <Maps
            address={address}
            crimeData={showCrimeLayer ? crimeCityData : []}
          />
        </div>
      </div>

      <div className='p-4 bg-white border-t border-gray-200'>
        <div className='flex justify-between items-center'>
          <div className='flex gap-2'>
            <button
              onClick={startProcessing}
              disabled={isProcessing}
              className={`px-4 py-2 rounded ${
                isProcessing
                  ? 'bg-gray-300'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              Start Processing
            </button>
            <button
              onClick={stopProcessing}
              disabled={!isProcessing}
              className={`px-4 py-2 rounded ${
                !isProcessing
                  ? 'bg-gray-300'
                  : 'bg-red-500 text-white hover:bg-red-600'
              }`}
            >
              Stop Processing
            </button>
          </div>

          <div className='text-sm text-gray-500'>
            Total Alerts:{' '}
            <span className='font-medium text-red-600'>
              {videoResults.length}
            </span>{' '}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Page;
