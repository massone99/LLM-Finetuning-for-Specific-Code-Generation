import akka.actor.{Actor, ActorSystem, Props, OneForOneStrategy, SupervisorStrategy, Terminator}

// Define the FaultyChildActor
class FaultyChildActor extends Actor {
  def receive: Receive = {
    case "causeNull" =>
      throw new NullPointerException("Child actor received null cause")
    case msg =>
      println(s"ChildActor received message: $msg")
  }
}

// Define the EscalatingSupervisor
class EscalatingSupervisor extends Actor with SupervisorStrategy {
  override def oneForOneStrategy(resolutionStrategy: SupervisorStrategy, exception: Exception): SupervisorStrategy = {
    // Log the exception before escalation
    println(s"EscalatingSupervisor received exception: ${exception.getClass.getName()} with message: ${exception.getMessage()}")
    // Escalate to the supervisor supervisor
    resolutionStrategy.withDeadLettersQueue()
  }
}

// Usage Example (for testing purposes)
object SupervisionApp extends App {
  val system = ActorSystem("SupervisionSystem")
  
  val supervisor = system.actorOf(Props(new EscalatingSupervisor()), "escalatingSupervisor")
  
  val childActor = supervisor.actorOf(Props(new FaultyChildActor()), "faultyChildActor")
  
  // Send a message to cause the failure
  childActor! "causeNull"
  
  // Allow some time for processing before shutdown
  Thread.sleep(1000)
  system.terminate()
}