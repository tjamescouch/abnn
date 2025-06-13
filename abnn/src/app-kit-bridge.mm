//
//  app-kit-bridge.mm
//  MetalNeuron
//
//  Created by James Couch on 2025-03-12.
//
#import <functional>
#import <Cocoa/Cocoa.h>
#include "app-kit-bridge.h"

static std::function<void()> trainNetworkHandler;
static std::function<void()> saveParamsHandler;
static std::function<void()> loadParamsHandler;

// Objective-C action handlers
@interface AppMenuHandler : NSObject
@end

@implementation AppMenuHandler
- (void)trainNetwork:(id)sender {
    if (trainNetworkHandler) trainNetworkHandler();
}

- (void)saveParameters:(id)sender {
    if (saveParamsHandler) saveParamsHandler();
}

- (void)loadParameters:(id)sender {
    if (loadParamsHandler) loadParamsHandler();
}

- (BOOL)validateUserInterfaceItem:(id<NSValidatedUserInterfaceItem>)item {
    return YES; // explicitly enables all menu items
}

@end

void setMenuActionHandlers(
    std::function<void()> run,
    std::function<void()> save,
    std::function<void()> load
) {
    trainNetworkHandler = run;
    saveParamsHandler = save;
    loadParamsHandler = load;
}

void setupMenus() {
    NSMenu* mainMenu = [[NSMenu alloc] init];

    // Application menu
    NSMenuItem* appMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:appMenuItem];

    NSMenu* appMenu = [[NSMenu alloc] initWithTitle:@"App"];
    NSString* appName = [[NSProcessInfo processInfo] processName];
    NSString* quitTitle = [@"Quit " stringByAppendingString:appName];
    NSMenuItem* quitMenuItem = [[NSMenuItem alloc] initWithTitle:quitTitle
                                                          action:@selector(terminate:)
                                                   keyEquivalent:@"q"];
    [appMenu addItem:quitMenuItem];
    [appMenuItem setSubmenu:appMenu];

    // Custom Actions menu
    NSMenuItem* actionsMenuItem = [[NSMenuItem alloc] init];
    [mainMenu addItem:actionsMenuItem];

    NSMenu* actionsMenu = [[NSMenu alloc] initWithTitle:@"Actions"];

    // Make this static so it persists
    static AppMenuHandler* handler = [[AppMenuHandler alloc] init];

    NSMenuItem* trainMenuItem = [[NSMenuItem alloc] initWithTitle:@"Run Network"
                                                           action:@selector(trainNetwork:)
                                                    keyEquivalent:@"l"];
    [trainMenuItem setTarget:handler];
    [actionsMenu addItem:trainMenuItem];

    NSMenuItem* saveParamsMenuItem = [[NSMenuItem alloc] initWithTitle:@"Save Parameters"
                                                                action:@selector(saveParameters:)
                                                         keyEquivalent:@"s"];
    [saveParamsMenuItem setTarget:handler];
    [actionsMenu addItem:saveParamsMenuItem];

    NSMenuItem* loadParamsMenuItem = [[NSMenuItem alloc] initWithTitle:@"Load Parameters"
                                                                action:@selector(loadParameters:)
                                                         keyEquivalent:@"o"];
    [loadParamsMenuItem setTarget:handler];
    [actionsMenu addItem:loadParamsMenuItem];

    [actionsMenuItem setSubmenu:actionsMenu];

    [NSApp setMainMenu:mainMenu];
}

static NSTextView* globalTextView = nil;

extern "C" void setupTextField(void* nsWindow) {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSWindow* window = (__bridge NSWindow*)nsWindow;

        NSRect frame = [window.contentView bounds];

        NSScrollView* scrollView = [[NSScrollView alloc] initWithFrame:frame];
        [scrollView setHasVerticalScroller:YES];
        [scrollView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];

        globalTextView = [[NSTextView alloc] initWithFrame:[[scrollView contentView] bounds]];
        [globalTextView setEditable:NO];
        [globalTextView setSelectable:YES];

        // Explicitly set text attributes:
        NSMutableParagraphStyle* paragraphStyle = [[NSMutableParagraphStyle alloc] init];
        [paragraphStyle setLineSpacing:1.0]; // Adds spacing between lines

        [globalTextView setTypingAttributes:@{
            NSFontAttributeName: [NSFont fontWithName:@"Menlo" size:10],
            NSParagraphStyleAttributeName: paragraphStyle,
            NSForegroundColorAttributeName: [NSColor textColor]
        }];

        [globalTextView setBackgroundColor:[NSColor textBackgroundColor]];
        [globalTextView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];

        [scrollView setDocumentView:globalTextView];
        [window.contentView addSubview:scrollView];
        
        [window makeFirstResponder:globalTextView];
    });
}

extern "C" void updateTextField(const char* message) {
    if (globalTextView) {
        const char* safeMsg = message ? message : "";
        NSString* safeString = [NSString stringWithUTF8String:safeMsg];
        if (!safeString) safeString = @"";

        dispatch_async(dispatch_get_main_queue(), ^{
            NSMutableParagraphStyle* paragraphStyle = [[NSMutableParagraphStyle alloc] init];
            [paragraphStyle setLineSpacing:0.5];

            NSDictionary* attributes = @{
                NSForegroundColorAttributeName : [NSColor textColor],
                NSFontAttributeName : [NSFont fontWithName:@"Menlo" size:11],
                NSParagraphStyleAttributeName : paragraphStyle
            };

            NSAttributedString* attributedMessage = [[NSAttributedString alloc]
                initWithString:[safeString stringByAppendingString:@""]
                attributes:attributes];

            NSScrollView* scrollView = [globalTextView enclosingScrollView];
            NSRect visibleRect = [scrollView contentView].documentVisibleRect;
            NSRect documentRect = [[scrollView documentView] bounds];

            BOOL isAtBottom = NSMaxY(visibleRect) >= NSMaxY(documentRect) - 1.0;

            [[globalTextView textStorage] appendAttributedString:attributedMessage];

            if (isAtBottom) {
                [globalTextView scrollRangeToVisible:NSMakeRange([[globalTextView string] length], 0)];
            }
        });
    }
}
