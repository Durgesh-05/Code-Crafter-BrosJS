import React from 'react';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import {
  crimeByCity,
  crimeCategories,
  crimeTrends,
  weaponUsage,
  barOptions,
  lineOptions,
  pieOptions,
} from '@/app/constant';

Chart.register(...registerables);

const CrimeCharts = () => {
  return (
    <div className='p-6 bg-amber-100 text-white min-h-screen'>
      <h1 className='text-3xl font-bold text-center mb-10 text-green-600 mt-4'>
        Crime Statistics Dashboard
      </h1>
      <div className='grid grid-cols-1 md:grid-cols-2 gap-10 mx-10'>
        <div className='bg-white p-4 rounded-3xl shadow-lg'>
          <h2 className='text-xl font-semibold mb-4 text-green-600'>
            Crime Count by City
          </h2>
          <div className='h-64'>
            <Bar data={crimeByCity} options={barOptions} />
          </div>
        </div>

        <div className='bg-white p-4 rounded-lg shadow-lg'>
          <h2 className='text-xl font-semibold mb-4 text-green-600'>
            Crime Trends Over Time
          </h2>
          <div className='h-64'>
            <Line data={crimeTrends} options={lineOptions} />
          </div>
        </div>
      </div>

      <div className='grid grid-cols-1 md:grid-cols-2 mt-8 gap-10 mx-10'>
        <div className='bg-white p-4 rounded-lg shadow-lg'>
          <h2 className='text-xl font-semibold mb-4 text-green-600'>
            Crime Category Distribution
          </h2>
          <div className='h-64'>
            <Pie data={crimeCategories} options={pieOptions} />
          </div>
        </div>

        <div className='bg-white p-4 rounded-lg shadow-lg'>
          <h2 className='text-xl font-semibold mb-4 text-green-600'>
            Weapon Usage Statistics
          </h2>
          <div className='h-64'>
            <Bar data={weaponUsage} options={barOptions} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default CrimeCharts;
