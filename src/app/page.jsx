'use client';
import React from 'react';
import { motion } from 'framer-motion';
import { Cctv, MapPinCheck, Siren } from 'lucide-react';
import { useRouter } from 'next/navigation';

const HeroSection = () => {
  const router = useRouter();

  return (
    <div className='bg-amber-100 text-emerald-600 min-h-screen flex flex-col items-center justify-center px-4 text-center relative font-mono'>
      {/* <SplashCursor /> */}
      <nav className='absolute top-0 w-full flex justify-between items-center p-6 max-w-7xl mx-auto'>
        <h1 className='text-lg font-semibold'>Suraksha-Ai</h1>
        <div className='flex space-x-6'>
          <div className='space-x-4'>
            <button
              className='font-semibold text-green-600 shadow-2xl bg-amber-200 px-4 py-2 rounded-xl hover:cursor-pointer'
              onClick={() => {
                router.push('/dashboard');
              }}
            >
              Dashboard
            </button>
          </div>
          <div className='space-x-4'>
            <button
              className='font-semibold text-green-600 shadow-2xl bg-amber-200 px-4 py-2 rounded-xl hover:cursor-pointer'
              onClick={() => {
                router.push('/summary');
              }}
            >
              Summary
            </button>
          </div>
          <div className='space-x-4'>
            <button
              className='font-semibold text-green-600 shadow-2xl bg-amber-200 px-4 py-2 rounded-xl hover:cursor-pointer'
              onClick={() => {
                router.push('/analytics');
              }}
            >
              Analytics
            </button>
          </div>
        </div>
      </nav>

      <div className='mt-48 text-center mb-28'>
        <motion.span
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className='bg-emerald-600 text-gray-200 px-4 py-2 rounded-full text-sm my-8 shadow-[0_3px_10px_rgb(0,0,0,0.2)] hover:cursor-pointer'
        >
          Introducing Suraksha-Ai 🚀
        </motion.span>
        <h2 className='text-6xl font-bold mt-14'>
          Autonomous Surveillance System <br />
          for Crime Detection
        </h2>
        <p className='text-gray-700 mt-4 max-w-2xl mx-auto'>
          Detects based on Live Camera Feed
        </p>
        <div className='mt-6 space-x-4'>
          {/* <SignedOut> */}
          <button
            className='bg-white text-black px-6 py-3 rounded-full font-medium hover:bg-green-600 hover:text-white shadow-[0_3px_10px_rgb(0,0,0,0.2)] hover:cursor-pointer hover:scale-105 transition-all duration-200'
            onClick={() => {
              router.push('/dashboard');
            }}
          >
            Get Started
          </button>
          {/* </SignedOut> */}
          {/* <SignedIn > */}
          {/* <UserButton /> */}
          {/* </SignedIn> */}
        </div>
      </div>

      <div className='w-full border-t border-gray-800 my-12'></div>

      {/* Text Feature Section */}
      <motion.div
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className='max-w-4xl text-center mt-28 mb-12'
      >
        <h3 className='text-3xl font-semibold mb-4'>Why Choose Suraksah-Ai?</h3>
        <p className='text-gray-700 text-lg'>
          An Autonomous Crime Detection with Camera Sensor Enabled along with
          Real-Time Alerts for Crime Prevention
        </p>
      </motion.div>

      {/* Feature Section */}
      <div className='mt-24 max-w-6xl w-full flex space-x-6 px-6 pb-12'>
        {[
          {
            icon: <Cctv className='text-gray-700 w-10 h-10 mx-auto' />,
            title: 'Crime Detection',
            desc: 'Real Time Detection of Crime and sending alerts to nearby police station.',
          },
          {
            icon: <Siren className='text-gray-700 w-10 h-10 mx-auto' />,
            title: 'Sos Alerts',
            desc: 'SOS Alerts for faster Response for the victims in need.',
          },
          {
            icon: <MapPinCheck className='text-gray-700 w-10 h-10 mx-auto' />,
            title: 'Map Alerts',
            desc: 'Map Based Pinned Real time Alerts.',
          },
        ].map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.2 }}
            viewport={{ once: true }}
            className='bg-white p-8 rounded-2xl text-center min-w-[300px] shadow-[0_3px_10px_rgb(0,0,0,0.2)] hover:scale-105 transition-all duration-200'
          >
            <div className='mb-4'>{feature.icon}</div>
            <h3 className='text-2xl gray-300 font-semibold'>{feature.title}</h3>
            <p className='text-gray-700 mt-3 text-lg'>{feature.desc}</p>
          </motion.div>
        ))}
      </div>

      {/* Footer */}
      <footer className='mt-24 w-full py-6 text-center border-t border-gray-800 text-gray-700 text-lg'>
        &copy; 2025 Suraksah-Ai. All rights reserved.
      </footer>

      {/* Sign-In Modal */}
      {/* <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full relative">
            <button 
              className="absolute top-2 right-2 text-black text-xl"
            >
              ✖
            </button> */}
      {/* <SignIn /> */}
      {/* </div> */}
      {/* </div> */}
      {/* )  */}
    </div>
  );
};

export default HeroSection;
