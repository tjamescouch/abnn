//
//  app-kit-bridge.h
//  MetalNeuron
//
//  Created by James Couch on 2025-03-12.
//

#ifndef APP_KIT_BRIDGE_H
#define APP_KIT_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

void setupTextField(void* nsWindow);
void updateTextField(const char* message);
void setupMenus();

void setMenuActionHandlers(
    std::function<void()> run,
    std::function<void()> save,
    std::function<void()> load
);

#ifdef __cplusplus
}
#endif

#endif
