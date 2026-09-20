import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ClerkProvider } from '@clerk/clerk-react'
import './index.css'
import App from './App.tsx'

const clerkKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {clerkKey ? <ClerkProvider publishableKey={clerkKey}><App /></ClerkProvider> : <div className="min-h-screen bg-slate-950 text-slate-200 grid place-items-center p-6 text-center"><div><h1 className="text-xl font-semibold">Authentication is not configured</h1><p className="mt-2 text-slate-400">Set VITE_CLERK_PUBLISHABLE_KEY before deploying MachinaSense.</p></div></div>}
  </StrictMode>,
)
