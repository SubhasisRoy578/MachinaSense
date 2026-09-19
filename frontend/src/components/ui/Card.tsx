import React from 'react';
import { cn } from '../../utils/cn';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {}

export function Card({ className, ...props }: CardProps) {
  return (
    <div 
      className={cn("industrial-card flex flex-col relative overflow-hidden", className)} 
      {...props} 
    />
  );
}

export function CardHeader({ className, ...props }: CardProps) {
  return (
    <div 
      className={cn("px-4 py-4 border-b border-slate-700/50 flex flex-row items-center justify-between", className)} 
      {...props} 
    />
  );
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 
      className={cn("text-sm font-semibold tracking-wide text-primary uppercase", className)} 
      {...props} 
    />
  );
}

export function CardContent({ className, ...props }: CardProps) {
  return (
    <div 
      className={cn("p-4 flex-1 flex flex-col", className)} 
      {...props} 
    />
  );
}
