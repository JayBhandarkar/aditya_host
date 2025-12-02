'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useLanguage } from '../contexts/LanguageContext'

interface SidebarProps {
  activeTab: string
  setActiveTab: (tab: string) => void
}

export default function Sidebar({ activeTab, setActiveTab }: SidebarProps) {
  const pathname = usePathname()
  const { t } = useLanguage()
  
  const menuItems = [
    { id: 'text-translator', label: t('nav.translate'), icon: '📝', href: '/text-translator' },
    { id: 'image-upload', label: 'Image/PDF Upload', icon: '🖼️', href: '/image-upload' },
    { id: 'learning-modules', label: t('nav.learning'), icon: '📚', href: '/learning-modules' },
    { id: 'literature-centre', label: 'Literature Centre', icon: '📖', href: '/literature-centre' },
    { id: 'get-extension', label: 'Get Extension', icon: '🚀', href: '/download-extension' },
  ]

  return (
    <>
      {menuItems.map((item) => (
        <Link key={item.id} href={item.href}>
          <div
            className={`sidebar-item ${pathname === item.href ? 'active' : ''}`}
          >
            <span className="sidebar-icon">{item.icon}</span>
            <span className="sidebar-text">{item.label}</span>
          </div>
        </Link>
      ))}
    </>
  )
}