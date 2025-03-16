import React from 'react';

export const crimeByCity = {
  labels: [
    'Delhi',
    'Mumbai',
    'Bangalore',
    'Hyderabad',
    'Kolkata',
    'Chennai',
    'Pune',
    'Ahmedabad',
    'Jaipur',
    'Lucknow',
  ],
  datasets: [
    {
      label: 'Crime Count',
      data: [5400, 4415, 3588, 2881, 2518, 2493, 2212, 1817, 1479, 1456],
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)', // Red
        'rgba(54, 162, 235, 0.6)', // Blue
        'rgba(255, 206, 86, 0.6)', // Yellow
        'rgba(75, 192, 192, 0.6)', // Teal
        'rgba(153, 102, 255, 0.6)', // Purple
        'rgba(255, 159, 64, 0.6)', // Orange
        'rgba(199, 199, 199, 0.6)', // Gray
        'rgba(83, 102, 255, 0.6)', // Dark Blue
        'rgba(255, 99, 255, 0.6)', // Pink
        'rgba(132, 99, 255, 0.6)', // Violet
      ],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
        'rgba(255, 159, 64, 1)',
        'rgba(199, 199, 199, 1)',
        'rgba(83, 102, 255, 1)',
        'rgba(255, 99, 255, 1)',
        'rgba(132, 99, 255, 1)',
      ],
      borderWidth: 1,
      borderRadius: 20, // Rounded top bars
    },
  ],
};

export const crimeTrends = {
  labels: [
    '2020-01',
    '2020-02',
    '2020-03',
    '2020-04',
    '2020-05',
    '2020-06',
    '2020-07',
    '2020-08',
    '2020-09',
    '2020-10',
  ],
  datasets: [
    {
      label: 'Crime Cases',
      data: [744, 696, 744, 720, 744, 720, 744, 744, 720, 744],
      borderColor: '#36A2EB',
      backgroundColor: 'rgba(54, 162, 235, 0.2)', // Light blue fill for smoothness
      fill: true, // Enable area filling
      pointBackgroundColor: '#fff', // White points
      pointBorderColor: '#36A2EB', // Blue border around points
      pointRadius: 6, // Bigger data points
      pointHoverRadius: 8, // Increase on hover
      borderWidth: 2, // Thicker line
      tension: 0.4, // Smooth curve effect
    },
  ],
};
export const crimeCategories = {
  labels: ['Other Crime', 'Violent Crime', 'Fire Accident', 'Traffic Fatality'],
  datasets: [
    {
      data: [22948, 11472, 3825, 1915],
      backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'],
      borderColor: '#fff', // White border for separation
      borderWidth: 2, // Clear separation between slices
      hoverOffset: 10, // Pops out segment on hover
      hoverBorderWidth: 3, // Increases the hover effect
    },
  ],
};

export const weaponUsage = {
  labels: ['Knife', 'None', 'Explosives', 'Blunt Object', 'Poison'],
  datasets: [
    {
      label: 'Usage Count',
      data: [5835, 5790, 5751, 5737, 5728],
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)', // Red
        'rgba(54, 162, 235, 0.6)', // Blue
        'rgba(255, 206, 86, 0.6)', // Yellow
        'rgba(75, 192, 192, 0.6)', // Teal
        'rgba(153, 102, 255, 0.6)', // Purple
      ],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(54, 162, 235, 1)',
        'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(153, 102, 255, 1)',
      ],
      borderWidth: 1,
      borderRadius: 20, // Gives the top of bars a rounded effect
    },
  ],
};

// Chart options with consistent height settings
export const barOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false, // Removed legend from bar charts
    },
  },
  scales: {
    x: {
      ticks: {
        font: {
          size: 10,
        },
      },
    },
  },
};

export const lineOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        font: {
          size: 14,
        },
        color: '#666',
        usePointStyle: true,
        pointStyle: 'circle',
      },
    },
  },
};

export const pieOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'bottom',
      labels: {
        font: {
          size: 14,
        },
        color: '#666',
        usePointStyle: true,
        pointStyle: 'circle',
      },
    },
  },
  layout: {
    padding: 10,
  },
};
